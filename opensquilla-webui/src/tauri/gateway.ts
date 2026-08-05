/**
 * Gateway adapter — Tauri-backed replacement for the WebSocket RPC gateway.
 *
 * The existing frontend talks to the Python gateway over a WebSocket RPC
 * protocol (`src/lib/rpc.ts`): `rpc.call('sessions.list', params)` for
 * requests and `rpc.on('session.event.text_delta', handler)` for streaming.
 * Under Tauri the Rust runtime lives in-process, so requests become
 * `invoke('command', args)` and streaming events arrive on the three Tauri
 * event channels (`turn-event`, `stream-event`, `tool-event`) defined in
 * {@link ./events.ts}.
 *
 * This module exposes the same logical surface the chat composables already
 * consume — session lifecycle, message send, history, compaction — but routes
 * every call through Tauri. The Rust `tauri::command` names targeted here are:
 *
 *   invoke('send_message',    { sessionId, content, ... })   → SendMessageResult
 *   invoke('create_session',  { ... })                       → SessionInfo
 *   invoke('list_sessions',   { limit?, view? })             → SessionListResult
 *   invoke('get_session',     { id })                        → SessionInfo
 *   invoke('delete_session',  { id })                        → void
 *   invoke('compact_session', { id })                        → CompactionResult
 *   invoke('chat_history',    { sessionKey, ... })           → ChatHistoryResult
 *
 * Streaming is delivered via {@link ./events.ts} `listen('turn-event'|'stream-event'|'tool-event')`.
 *
 * Payload interfaces mirror the Rust serde structs (`SessionInfo`,
 * `SendMessageResult`, etc.) and reuse the existing per-event payload types
 * from `src/types/rpc.ts` where the wire shape is unchanged so chat
 * composables can adopt this adapter without rewriting their event handling.
 */

import { invoke, invokeCommand } from './invoke'
import {
  listen,
  type StreamHandlers,
  type TauriEventListener,
  type TurnEventPayload,
  type StreamEventPayload,
  type ToolEventPayload,
  type UnlistenFn,
} from './events'

/* ──────────────────────────────────────────────────────────────────────────
 * Shared types. These mirror the Rust serde structs returned by the gateway
 * commands. snake_case fields are kept alongside camelCase aliases because the
 * Rust runtime emits serde's default snake_case while the existing renderer
 * was written against the gateway's mixed-convention payloads (see
 * src/types/rpc.ts RawSessionItem). Keeping both lets us swap transports
 * without churning every composable.
 * ────────────────────────────────────────────────────────────────────────── */

export interface SessionInfo {
  id: string
  key?: string
  session_key?: string
  sessionKey?: string
  title?: string
  subject?: string
  subtitle?: string
  surface?: string
  agentId?: string
  agent_id?: string
  effectiveAgentId?: string
  workspaceId?: string
  workspace_id?: string
  updatedAt?: number | string
  updated_at?: number | string
  lastActivityAt?: number | string
  last_activity_at?: number | string
  messageCount?: number
  message_count?: number
  status?: string
  runStatus?: string
  run_status?: string
  model?: string
  [key: string]: unknown
}

export interface SessionListResult {
  sessions?: SessionInfo[]
  keys?: string[]
}

export interface CreateSessionParams {
  /** Agent to bind to the session. */
  agentId?: string
  agent_id?: string
  /** Display title hint. */
  title?: string
  /** Project workspace id. */
  workspaceId?: string
  workspace_id?: string
  /** Conversation kind (chat / task / etc.). */
  kind?: string
  /** Fork parent session key, when forking. */
  forkFrom?: string
  fork_from?: string
  [key: string]: unknown
}

export interface SendMessageAttachment {
  type: string
  mime: string
  name: string
  data?: string
  file_uuid?: string
  fileUuid?: string
}

export interface SendMessageParams {
  /** Target session id (Tauri commands use camelCase keys). */
  sessionId: string
  /** User message content. */
  content: string
  /** Stable idempotency key for one logical send attempt. */
  clientRequestId?: string
  /** Stable client identity for reconciling the optimistic user row. */
  clientMessageId?: string
  /** Display text override (when content is a slash command). */
  displayText?: string
  display_text?: string
  /** Run mode for this turn. */
  runMode?: 'standard' | 'trusted' | 'full'
  run_mode?: 'standard' | 'trusted' | 'full'
  /** Whether this send was elevated (e.g. confirm-flow). */
  elevated?: string
  /** Collaboration mode (plan mode). */
  collaborationMode?: string
  collaboration_mode?: string
  /** Attachments carried with the message. */
  attachments?: SendMessageAttachment[]
  [key: string]: unknown
}

export interface SendMessageResult {
  sessionId?: string
  session_id?: string
  sessionKey?: string
  session_key?: string
  messageId?: string
  message_id?: string
  userMessageId?: string
  user_message_id?: string
  clientMessageId?: string
  client_message_id?: string
  taskId?: string
  task_id?: string
  taskStatus?: string
  task_status?: string
  replayed?: boolean
  terminalReason?: string
  terminal_reason?: string
  terminalMessage?: string
  terminal_message?: string
  reason?: string
}

export interface ChatHistoryParams {
  sessionKey: string
  session_key?: string
  /** Cursor for pagination (older direction). */
  before?: string | number | null
  limit?: number
  /** Whether to include compaction summaries. */
  includeCompaction?: boolean
  include_compaction?: boolean
  [key: string]: unknown
}

export interface ChatHistoryMessage {
  role?: string
  text?: string
  timestamp?: string | number | null
  ts?: string | number | null
  id?: string
  message_id?: string
  messageId?: string
  attachments?: unknown[]
  artifacts?: unknown[]
  tool_calls?: unknown[]
  toolCalls?: unknown[]
  timeline?: unknown[]
  reasoning_content?: string
  reasoningContent?: string
  usage?: unknown
  model?: string
  [key: string]: unknown
}

export interface ChatCompactionSummary {
  id?: string | number | null
  compactionId?: string | null
  compaction_id?: string | null
  compactionIndex?: number | null
  compaction_index?: number | null
  triggerReason?: string | null
  trigger_reason?: string | null
  summaryText?: string
  summary_text?: string
  coverageStatus?: string
  coverage_status?: string
  removedCount?: number | null
  removed_count?: number | null
  keptCount?: number | null
  kept_count?: number | null
  createdAt?: string | number | null
  created_at?: string | number | null
}

export interface ChatHistoryResult {
  messages?: ChatHistoryMessage[]
  hasMore?: boolean
  has_more?: boolean
  oldestCursor?: string | number | null
  oldest_cursor?: string | number | null
  newestCursor?: string | number | null
  newest_cursor?: string | number | null
  historyScope?: string
  history_scope?: string
  canonicalAvailable?: boolean
  canonical_available?: boolean
  canonicalComplete?: boolean
  canonical_complete?: boolean
  limit?: number
  returned?: number
  compactionSummaries?: ChatCompactionSummary[]
  compaction_summaries?: ChatCompactionSummary[]
  turnOutcomes?: unknown[]
  turn_outcomes?: unknown[]
}

export interface CompactionResult {
  sessionId?: string
  session_id?: string
  sessionKey?: string
  session_key?: string
  status?: string
  compacted?: boolean
  compactionId?: string
  compaction_id?: string
  detail?: string
  removedCount?: number
  removed_count?: number
  keptCount?: number
  kept_count?: number
}

export interface DeleteSessionResult {
  ok?: boolean
  deleted?: string[]
  errors?: Array<{ key?: string; error?: string }>
  [key: string]: unknown
}

/* ──────────────────────────────────────────────────────────────────────────
 * Request surface. Each function maps 1:1 to a Rust tauri::command.
 * ────────────────────────────────────────────────────────────────────────── */

/**
 * Send a user message and start a turn. Returns once the turn is queued; the
 * streaming response (text deltas, tool calls, state changes) arrives over the
 * `turn-event` / `stream-event` / `tool-event` channels — subscribe with
 * {@link subscribeToSession} or the raw `listen` helpers.
 *
 * Rust: `#[tauri::command] async fn send_message(session_id: String, content: String, ...) -> Result<SendMessageResult, IpcError>`.
 */
export async function sendMessage(
  sessionId: string,
  content: string,
  options: Omit<SendMessageParams, 'sessionId' | 'content'> = {},
): Promise<SendMessageResult> {
  return invoke<SendMessageResult>('send_message', {
    sessionId,
    content,
    ...options,
  })
}

/**
 * Create a new session. Rust: `create_session(params: CreateSessionParams) -> Result<SessionInfo, IpcError>`.
 */
export async function createSession(
  params: CreateSessionParams = {},
): Promise<SessionInfo> {
  return invoke<SessionInfo>('create_session', params)
}

/**
 * List sessions. Mirrors the `sessions.list` RPC. Rust:
 * `list_sessions(limit: Option<usize>, view: Option<String>) -> Result<SessionListResult, IpcError>`.
 */
export async function listSessions(
  options: { limit?: number; view?: string } = {},
): Promise<SessionListResult> {
  return invoke<SessionListResult>('list_sessions', {
    limit: options.limit,
    view: options.view,
  })
}

/**
 * Get a single session by id. Mirrors the conceptual `sessions.get` (the
 * legacy gateway composed this from `sessions.preview`; under Tauri the Rust
 * side exposes it directly). Rust: `get_session(id: String) -> Result<SessionInfo, IpcError>`.
 */
export async function getSession(id: string): Promise<SessionInfo> {
  return invoke<SessionInfo>('get_session', { id })
}

/**
 * Delete a session. Rust: `delete_session(id: String) -> Result<DeleteSessionResult, IpcError>`.
 * For batch deletes (the legacy `sessions.delete` accepted `keys: string[]`),
 * call this per key or use {@link deleteSessions}.
 */
export async function deleteSession(id: string): Promise<DeleteSessionResult> {
  return invoke<DeleteSessionResult>('delete_session', { id })
}

/** Batch delete helper. Issues parallel `delete_session` calls. */
export async function deleteSessions(
  ids: string[],
): Promise<DeleteSessionResult[]> {
  return Promise.all(ids.map((id) => deleteSession(id)))
}

/**
 * Compact a session's context window. Mirrors the legacy compaction flow.
 * Rust: `compact_session(id: String) -> Result<CompactionResult, IpcError>`.
 */
export async function compactSession(id: string): Promise<CompactionResult> {
  return invoke<CompactionResult>('compact_session', { id })
}

/**
 * Fetch chat history for a session. Mirrors the `chat.history` RPC.
 * Rust: `chat_history(session_key: String, before: Option<...>, limit: Option<usize>) -> Result<ChatHistoryResult, IpcError>`.
 */
export async function getChatHistory(
  params: ChatHistoryParams,
): Promise<ChatHistoryResult> {
  // Normalize the session key into the camelCase field the Rust command reads.
  return invoke<ChatHistoryResult>('chat_history', {
    sessionKey: params.sessionKey ?? params.session_key,
    before: params.before,
    limit: params.limit,
    includeCompaction: params.includeCompaction ?? params.include_compaction,
  })
}

/**
 * Abort the active turn for a session. Mirrors the `sessions.abort` RPC.
 * Rust: `abort_session(id: String) -> Result<AbortResult, IpcError>`.
 */
export async function abortSession(
  id: string,
): Promise<{ ok?: boolean; task_id?: string; taskId?: string }> {
  return invoke('abort_session', { id })
}

/**
 * Fork a session. Mirrors the `sessions.fork` RPC.
 * Rust: `fork_session(key: String) -> Result<{ key: String }, IpcError>`.
 */
export async function forkSession(key: string): Promise<{ key?: string }> {
  return invoke('fork_session', { key })
}

/* ──────────────────────────────────────────────────────────────────────────
 * Streaming subscription surface.
 *
 * The chat composables in src/composables/chat/useChatRpcSubscriptions.ts
 * today register ~20 granular `rpc.on('session.event.*')` handlers. Under
 * Tauri those collapse onto three channels. subscribeToSession() accepts a
 * handler bag keyed by the original discriminator so callers can keep their
 * per-event logic and just rewire the transport.
 * ────────────────────────────────────────────────────────────────────────── */

/**
 * Handlers keyed by the original gateway event discriminator. Any payload
 * whose `event` field matches is dispatched to the corresponding handler.
 * This lets `useChatRpcSubscriptions` swap `rpc.on(name, h)` for a single
 * `subscribeToSession(key, handlers)` call.
 */
export interface SessionEventHandlers {
  /** `session.event.text_delta` */
  onTextDelta?: TauriEventListener<StreamEventPayload>
  /** `session.event.tool_use_start` */
  onToolUseStart?: TauriEventListener<ToolEventPayload>
  /** `session.event.tool_use_delta` */
  onToolUseDelta?: TauriEventListener<ToolEventPayload>
  /** `session.event.tool_result` */
  onToolResult?: TauriEventListener<ToolEventPayload>
  /** `session.event.artifact` */
  onArtifact?: TauriEventListener<StreamEventPayload>
  /** `session.event.router_decision` */
  onRouterDecision?: TauriEventListener<StreamEventPayload>
  /** `session.event.ensemble_progress` */
  onEnsembleProgress?: TauriEventListener<StreamEventPayload>
  /** `session.event.state_change` */
  onStateChange?: TauriEventListener<TurnEventPayload>
  /** `session.event.run_heartbeat` */
  onRunHeartbeat?: TauriEventListener<TurnEventPayload>
  /** `session.event.compaction` */
  onCompaction?: TauriEventListener<TurnEventPayload>
  /** `session.event.warning` */
  onWarning?: TauriEventListener<TurnEventPayload>
  /** `session.event.input_disposition` */
  onInputDisposition?: TauriEventListener<TurnEventPayload>
  /** `session.event.cron_result` */
  onCronResult?: TauriEventListener<TurnEventPayload>
  /** `session.event.subagent_completion` */
  onSubagentCompletion?: TauriEventListener<TurnEventPayload>
  /** `session.epoch_changed` */
  onEpochChanged?: TauriEventListener<TurnEventPayload>
  /** Catch-all for any other discriminator. */
  onAny?: (channel: 'turn' | 'stream' | 'tool', payload: TurnEventPayload | StreamEventPayload | ToolEventPayload) => void
}

/**
 * Subscribe to all streaming events for a session. Returns a single unlisten
 * function that tears down the underlying Tauri listeners.
 *
 * Events are filtered by `session_key`/`sessionKey` when the param is
 * provided, so a multi-session view can subscribe narrowly. Events without a
 * session key pass through unfiltered (the runtime emits some global
 * lifecycle events this way).
 *
 * @param sessionKey - Optional session key to filter on.
 * @param handlers   - Discriminator-keyed handler bag.
 */
export function subscribeToSession(
  sessionKey: string | null,
  handlers: SessionEventHandlers,
): UnlistenFn {
  const matchesSession = (p: { session_key?: string; sessionKey?: string }): boolean => {
    if (!sessionKey) return true
    return p.session_key === sessionKey || p.sessionKey === sessionKey
  }

  const onTurn = (payload: TurnEventPayload): void => {
    if (!matchesSession(payload)) return
    switch (payload.event) {
      case 'session.event.state_change': handlers.onStateChange?.(payload); break
      case 'session.event.run_heartbeat': handlers.onRunHeartbeat?.(payload); break
      case 'session.event.compaction': handlers.onCompaction?.(payload); break
      case 'session.event.warning': handlers.onWarning?.(payload); break
      case 'session.event.input_disposition': handlers.onInputDisposition?.(payload); break
      case 'session.event.cron_result': handlers.onCronResult?.(payload); break
      case 'session.event.subagent_completion': handlers.onSubagentCompletion?.(payload); break
      case 'session.epoch_changed': handlers.onEpochChanged?.(payload); break
      default: break
    }
    handlers.onAny?.('turn', payload)
  }

  const onStream = (payload: StreamEventPayload): void => {
    if (!matchesSession(payload)) return
    switch (payload.event) {
      case 'session.event.text_delta': handlers.onTextDelta?.(payload); break
      case 'session.event.artifact': handlers.onArtifact?.(payload); break
      case 'session.event.router_decision': handlers.onRouterDecision?.(payload); break
      case 'session.event.ensemble_progress': handlers.onEnsembleProgress?.(payload); break
      default: break
    }
    handlers.onAny?.('stream', payload)
  }

  const onTool = (payload: ToolEventPayload): void => {
    if (!matchesSession(payload)) return
    switch (payload.event) {
      case 'session.event.tool_use_start': handlers.onToolUseStart?.(payload); break
      case 'session.event.tool_use_delta': handlers.onToolUseDelta?.(payload); break
      case 'session.event.tool_result': handlers.onToolResult?.(payload); break
      default: break
    }
    handlers.onAny?.('tool', payload)
  }

  const unlisteners: UnlistenFn[] = [
    listen<TurnEventPayload>('turn-event', onTurn),
    listen<StreamEventPayload>('stream-event', onStream),
    listen<ToolEventPayload>('tool-event', onTool),
  ]
  return () => {
    for (const u of unlisteners) u()
    unlisteners.length = 0
  }
}

/**
 * Subscribe to the three raw streaming channels without discriminator
 * dispatch. Useful for components that only care about the channel, not the
 * specific event. Auto-cleans up on component unmount.
 */
export function subscribeToStreams(handlers: StreamHandlers): { unlisten: UnlistenFn } {
  const unlisteners: UnlistenFn[] = []
  if (handlers.onTurnEvent) {
    unlisteners.push(listen<TurnEventPayload>('turn-event', handlers.onTurnEvent))
  }
  if (handlers.onStreamEvent) {
    unlisteners.push(listen<StreamEventPayload>('stream-event', handlers.onStreamEvent))
  }
  if (handlers.onToolEvent) {
    unlisteners.push(listen<ToolEventPayload>('tool-event', handlers.onToolEvent))
  }
  return {
    unlisten: () => {
      for (const u of unlisteners) u()
      unlisteners.length = 0
    },
  }
}

/** Re-exported invoke helper for gateway commands not yet wrapped above. */
export { invokeCommand }
