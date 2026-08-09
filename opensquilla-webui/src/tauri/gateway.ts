/**
 * Gateway adapter — Tauri-backed replacement for the WebSocket RPC gateway.
 *
 * The existing frontend talks to the Python gateway over a WebSocket RPC
 * protocol (`src/lib/rpc.ts`): `rpc.call('sessions.list', params)` for
 * requests and `rpc.on('session.event.text_delta', handler)` for streaming.
 * Under Tauri the Rust runtime lives in-process, so requests become
 * `invoke('command', args)` and streaming events arrive on the per-session
 * Tauri event channel `agent:stream:{sessionId}` (see {@link ./events.ts}).
 *
 * This module exposes the same logical surface the chat composables already
 * consume — session lifecycle, message send, history, compaction — but routes
 * every call through Tauri. The Rust `tauri::command` names targeted here are
 * (see `src-tauri/src/agent_bridge.rs` + `src-tauri/src/commands.rs`):
 *
 *   invoke('send_message',      { request: { sessionId, message, ... } }) → string (turn id)
 *   invoke('send_message_sync', { request: { ... } })                     → MessageSendResponse
 *   invoke('create_session',    { request: { title?, model?, ... } })     → { session }
 *   invoke('list_sessions',     { })                                      → { sessions, count }
 *   invoke('get_session',       { sessionId })                            → { session }
 *   invoke('delete_session',    { sessionId })                            → bool
 *   invoke('archive_session',   { sessionId })                            → { session }
 *   invoke('compact_session',   { sessionId })                            → { ok, message? }
 *   invoke('get_chat_history',  { sessionId })                            → MessageDto[]
 *   invoke('clear_chat_history',{ sessionId })                            → bool
 *   invoke('abort_session',     { sessionId })                            → bool
 *   invoke('fork_session',      { sessionId, forkEvent?, title? })        → { session }
 *
 * Streaming is delivered via {@link ./events.ts}
 * `listen('agent:stream:{sessionId}')`. The listen MUST be attached before
 * `send_message` is invoked so no early turn events are missed.
 */

import { invoke, invokeCommand } from './invoke'
import {
  agentStreamChannel,
  listen,
  type MessageDto,
  type MessageStreamEvent,
  type StreamHandlers,
  type StreamEventPayload,
  type TauriEventListener,
  type UnlistenFn,
} from './events'

/* ──────────────────────────────────────────────────────────────────────────
 * Shared types. These mirror the Rust serde structs returned by the gateway
 * commands (all `#[serde(rename_all = "camelCase")]`).
 * ────────────────────────────────────────────────────────────────────────── */

/** `SessionInfo` returned by the session commands. */
export interface SessionInfo {
  id: string
  title: string
  model: string
  agentId: string
  createdAt: string
  updatedAt: string
  state: string
  mode: string
  messageCount: number
  systemPrompt?: string | null
  totalTokens: number
  [key: string]: unknown
}

/** `SessionListResponse` — `{ sessions, count }`. */
export interface SessionListResult {
  sessions: SessionInfo[]
  count: number
}

/** `SessionResponse` — `{ session }`. */
export interface SessionResult {
  session: SessionInfo
}

/** Input for `create_session`. */
export interface CreateSessionParams {
  /** Display title hint. Defaults to "New Session". */
  title?: string
  /** Model to bind to the session. */
  model?: string
  /** Agent id to associate with the session. */
  agentId?: string
  /** Optional system prompt. */
  systemPrompt?: string
  /** Session mode: "chat" | "plan" | "agent" | "batch". */
  mode?: string
  [key: string]: unknown
}

/** Extra send options (beyond sessionId + message). */
export interface MessageSendParams {
  /** Optional conversation history (for stateless mode). */
  history?: MessageDto[]
  /** Override the model for this request. */
  model?: string
  /** Override the provider for this request. */
  provider?: string
  /** Whether to stream the response. Defaults to true. */
  stream?: boolean
}

/** `MessageSendResponse` from `send_message_sync`. */
export interface MessageSendResponse {
  turnId: string
  sessionId: string
  messages: MessageDto[]
  usage: { inputTokens: number; outputTokens: number; totalTokens: number }
  durationMs: number
}

/** `{ ok, message? }` returned by `compact_session` / `reset_config`. */
export interface OperationResult {
  ok: boolean
  message?: string | null
}

/** `{ session: SessionInfo }` returned by `fork_session`. */
export interface ForkSessionResult {
  session: SessionInfo
}

/** Result of a delete (a bare boolean from Rust). */
export type DeleteSessionResult = boolean

/* ──────────────────────────────────────────────────────────────────────────
 * Request surface. Each function maps 1:1 to a Rust tauri::command.
 * ────────────────────────────────────────────────────────────────────────── */

/**
 * Send a user message and start a turn. Returns the turn id once the turn is
 * queued; the streaming response arrives over `agent:stream:{sessionId}` —
 * subscribe with {@link subscribeToSession} or {@link subscribeToStreams}
 * BEFORE calling this.
 *
 * Rust: `#[tauri::command] async fn send_message(request: MessageSendRequest) -> Result<String, TauriError>`.
 */
export async function sendMessage(
  sessionId: string,
  message: string,
  options: MessageSendParams = {},
): Promise<string> {
  const request: Record<string, unknown> = {
    sessionId,
    message,
  }
  if (options.history !== undefined) request.history = options.history
  if (options.model !== undefined) request.model = options.model
  if (options.provider !== undefined) request.provider = options.provider
  if (options.stream !== undefined) request.stream = options.stream
  return invoke<string>('send_message', { request })
}

/**
 * Send a message synchronously (non-streaming) and return the full response.
 * Rust: `send_message_sync(request: MessageSendRequest) -> Result<MessageSendResponse, TauriError>`.
 */
export async function sendMessageSync(
  sessionId: string,
  message: string,
  options: MessageSendParams = {},
): Promise<MessageSendResponse> {
  const request: Record<string, unknown> = {
    sessionId,
    message,
  }
  if (options.history !== undefined) request.history = options.history
  if (options.model !== undefined) request.model = options.model
  if (options.provider !== undefined) request.provider = options.provider
  request.stream = options.stream ?? false
  return invoke<MessageSendResponse>('send_message_sync', { request })
}

/**
 * Create a new session. Rust:
 * `create_session(request: SessionCreateRequest) -> Result<SessionResponse, TauriError>`.
 */
export async function createSession(
  params: CreateSessionParams = {},
): Promise<SessionInfo> {
  const result = await invoke<SessionResult>('create_session', {
    request: {
      title: params.title,
      model: params.model,
      agentId: params.agentId,
      systemPrompt: params.systemPrompt,
      mode: params.mode,
    },
  })
  return result.session
}

/**
 * List sessions. Rust: `list_sessions() -> Result<SessionListResponse, TauriError>`.
 */
export async function listSessions(): Promise<SessionListResult> {
  return invoke<SessionListResult>('list_sessions', {})
}

/**
 * Get a single session by id. Rust: `get_session(session_id: String) -> Result<SessionResponse, TauriError>`.
 */
export async function getSession(id: string): Promise<SessionInfo> {
  const result = await invoke<SessionResult>('get_session', { sessionId: id })
  return result.session
}

/**
 * Delete a session. Rust: `delete_session(session_id: String) -> Result<bool, TauriError>`.
 * For batch deletes (the legacy `sessions.delete` accepted `keys: string[]`),
 * call this per key or use {@link deleteSessions}.
 */
export async function deleteSession(id: string): Promise<boolean> {
  return invoke<boolean>('delete_session', { sessionId: id })
}

/** Batch delete helper. Issues parallel `delete_session` calls. */
export async function deleteSessions(ids: string[]): Promise<boolean[]> {
  return Promise.all(ids.map((id) => deleteSession(id)))
}

/**
 * Archive a session. Rust: `archive_session(session_id: String) -> Result<SessionResponse, TauriError>`.
 */
export async function archiveSession(id: string): Promise<SessionInfo> {
  const result = await invoke<SessionResult>('archive_session', { sessionId: id })
  return result.session
}

/**
 * Compact a session's context window. Rust:
 * `compact_session(session_id: String) -> Result<OperationResult, TauriError>`.
 */
export async function compactSession(id: string): Promise<OperationResult> {
  return invoke<OperationResult>('compact_session', { sessionId: id })
}

/**
 * Fetch chat history for a session. Rust:
 * `get_chat_history(session_id: String) -> Result<Vec<MessageDto>, TauriError>` —
 * returns a BARE array of `MessageDto`.
 */
export async function getChatHistory(sessionId: string): Promise<MessageDto[]> {
  return invoke<MessageDto[]>('get_chat_history', { sessionId })
}

/**
 * Clear a session's chat history. Rust:
 * `clear_chat_history(session_id: String) -> Result<bool, TauriError>`.
 */
export async function clearChatHistory(sessionId: string): Promise<boolean> {
  return invoke<boolean>('clear_chat_history', { sessionId })
}

/**
 * Abort the active turn for a session. Rust:
 * `abort_session(session_id: String) -> Result<bool, TauriError>`.
 */
export async function abortSession(id: string): Promise<boolean> {
  return invoke<boolean>('abort_session', { sessionId: id })
}

/**
 * Fork a session. Rust:
 * `fork_session(session_id: String, fork_event: Option<...>, title: Option<String>) -> Result<SessionResponse, TauriError>`.
 */
export async function forkSession(
  id: string,
  options: { forkEvent?: unknown; title?: string } = {},
): Promise<ForkSessionResult> {
  const args: Record<string, unknown> = { sessionId: id }
  if (options.forkEvent !== undefined) args.forkEvent = options.forkEvent
  if (options.title !== undefined) args.title = options.title
  return invoke<ForkSessionResult>('fork_session', args)
}

/* ──────────────────────────────────────────────────────────────────────────
 * Streaming subscription surface.
 *
 * Under Tauri all turn/stream/tool events for a session arrive on the single
 * channel `agent:stream:{sessionId}` as a `MessageStreamEvent`. subscribeToSession()
 * accepts a handler bag keyed by the original gateway discriminator and maps
 * the `type` tag onto it; subscribeToStreams() dispatches the raw three-group
 * {@link StreamHandlers} bag.
 * ────────────────────────────────────────────────────────────────────────── */

/** Loose payload type for the granular (gateway-era) handlers. */
type GranularEventPayload = Record<string, unknown>

/**
 * Handlers keyed by the original gateway event discriminator. Any payload
 * whose `event` field matches is dispatched to the corresponding handler.
 * This lets `useChatRpcSubscriptions` swap `rpc.on(name, h)` for a single
 * `subscribeToSession(sessionId, handlers)` call.
 */
export interface SessionEventHandlers {
  /** `session.event.text_delta` */
  onTextDelta?: TauriEventListener<GranularEventPayload>
  /** `session.event.tool_use_start` */
  onToolUseStart?: TauriEventListener<GranularEventPayload>
  /** `session.event.tool_use_delta` */
  onToolUseDelta?: TauriEventListener<GranularEventPayload>
  /** `session.event.tool_result` */
  onToolResult?: TauriEventListener<GranularEventPayload>
  /** `session.event.artifact` */
  onArtifact?: TauriEventListener<GranularEventPayload>
  /** `session.event.router_decision` */
  onRouterDecision?: TauriEventListener<GranularEventPayload>
  /** `session.event.ensemble_progress` */
  onEnsembleProgress?: TauriEventListener<GranularEventPayload>
  /** `session.event.state_change` */
  onStateChange?: TauriEventListener<GranularEventPayload>
  /** `session.event.run_heartbeat` */
  onRunHeartbeat?: TauriEventListener<GranularEventPayload>
  /** `session.event.compaction` */
  onCompaction?: TauriEventListener<GranularEventPayload>
  /** `session.event.warning` */
  onWarning?: TauriEventListener<GranularEventPayload>
  /** `session.event.input_disposition` */
  onInputDisposition?: TauriEventListener<GranularEventPayload>
  /** `session.event.cron_result` */
  onCronResult?: TauriEventListener<GranularEventPayload>
  /** `session.event.subagent_completion` */
  onSubagentCompletion?: TauriEventListener<GranularEventPayload>
  /** `session.epoch_changed` */
  onEpochChanged?: TauriEventListener<GranularEventPayload>
  /** Catch-all for any other discriminator. */
  onAny?: (channel: 'turn' | 'stream' | 'tool', payload: unknown) => void
}

/** Map a `MessageStreamEvent` onto the granular gateway-era handlers. */
function dispatchStreamEvent(
  payload: MessageStreamEvent,
  handlers: SessionEventHandlers,
): void {
  if (!payload || typeof payload.type !== 'string') return
  switch (payload.type) {
    case 'stream_event': {
      const inner = payload.data
      if (!inner || typeof inner.event !== 'string') return
      switch (inner.event) {
        case 'content_block_delta': {
          const delta = inner.data.delta
          if (delta?.type === 'text_delta') {
            const data = { event: 'session.event.text_delta', text: delta.text }
            handlers.onTextDelta?.(data)
            handlers.onAny?.('stream', data)
          } else if (delta?.type === 'input_json_delta') {
            const data = {
              event: 'session.event.tool_use_delta',
              json_fragment: delta.partial_json,
              jsonFragment: delta.partial_json,
            }
            handlers.onToolUseDelta?.(data)
            handlers.onAny?.('stream', data)
          }
          break
        }
        case 'content_block_start': {
          const block = inner.data.block
          if (block?.type === 'tool_use') {
            const data = {
              event: 'session.event.tool_use_start',
              id: block.id,
              tool_use_id: block.id,
              toolUseId: block.id,
              name: block.name,
              input: block.input,
            }
            handlers.onToolUseStart?.(data)
            handlers.onAny?.('tool', data)
          }
          break
        }
        case 'error': {
          const data = {
            event: 'session.event.warning',
            message: inner.data.message,
            code: inner.data.code ?? 'STREAM_ERROR',
          }
          handlers.onWarning?.(data)
          handlers.onAny?.('turn', data)
          break
        }
        default:
          break
      }
      break
    }
    case 'tool_call_start': {
      const data = {
        event: 'session.event.tool_use_start',
        id: payload.data.tool_call_id,
        tool_use_id: payload.data.tool_call_id,
        toolUseId: payload.data.tool_call_id,
        name: payload.data.name,
        input: payload.data.input,
      }
      handlers.onToolUseStart?.(data)
      handlers.onAny?.('tool', data)
      break
    }
    case 'tool_call_complete': {
      const data = {
        event: 'session.event.tool_result',
        id: payload.data.tool_call_id,
        tool_use_id: payload.data.tool_call_id,
        toolUseId: payload.data.tool_call_id,
        name: payload.data.name,
        content: payload.data.content,
        is_error: payload.data.is_error,
        isError: payload.data.is_error,
      }
      handlers.onToolResult?.(data)
      handlers.onAny?.('tool', data)
      break
    }
    case 'turn_start': {
      const data = {
        event: 'session.event.state_change',
        status: 'running',
        turn_id: payload.data.turn_id,
        session_id: payload.data.session_id,
        timestamp: payload.data.timestamp,
      }
      handlers.onStateChange?.(data)
      handlers.onAny?.('turn', data)
      break
    }
    case 'generation_start': {
      const data = {
        event: 'session.event.state_change',
        status: 'generating',
        model: payload.data.model,
        provider: payload.data.provider,
      }
      handlers.onStateChange?.(data)
      handlers.onAny?.('turn', data)
      break
    }
    case 'turn_complete': {
      const data = {
        event: 'session.event.state_change',
        status: 'completed',
        turn_id: payload.data.turn_id,
        session_id: payload.data.session_id,
        messages: payload.data.messages,
        usage: payload.data.usage,
        duration_ms: payload.data.duration_ms,
      }
      handlers.onStateChange?.(data)
      handlers.onAny?.('turn', data)
      break
    }
    case 'turn_error': {
      const warning = { event: 'session.event.warning', message: payload.data.message, code: payload.data.code ?? 'TURN_ERROR' }
      const failed = {
        event: 'session.event.state_change',
        status: 'failed',
        turn_id: payload.data.turn_id,
        session_id: payload.data.session_id,
        message: payload.data.message,
        code: payload.data.code,
      }
      handlers.onWarning?.(warning)
      handlers.onStateChange?.(failed)
      handlers.onAny?.('turn', warning)
      break
    }
    case 'compaction': {
      const data = {
        event: 'session.event.compaction',
        before_count: payload.data.before_count,
        after_count: payload.data.after_count,
        success: payload.data.success,
      }
      handlers.onCompaction?.(data)
      handlers.onAny?.('turn', data)
      break
    }
    default:
      break
  }
}

/**
 * Subscribe to all streaming events for a session. Returns an unlisten
 * function that tears down the underlying Tauri listener.
 *
 * The listener is attached to `agent:stream:{sessionId}` and MUST be created
 * before `send_message` is invoked so the turn_start event is not missed.
 *
 * @param sessionId - Session whose stream to subscribe to.
 * @param handlers  - Discriminator-keyed handler bag.
 */
export function subscribeToSession(
  sessionId: string,
  handlers: SessionEventHandlers,
): UnlistenFn {
  const unlisten = listen<MessageStreamEvent>(agentStreamChannel(sessionId), (payload) => {
    dispatchStreamEvent(payload, handlers)
  })
  return () => {
    unlisten()
  }
}

/**
 * Subscribe to a session's streaming channel and dispatch the raw three-group
 * {@link StreamHandlers} bag (onTurnEvent / onStreamEvent / onToolEvent).
 * Auto-cleans up on component unmount.
 */
export function subscribeToStreams(
  sessionId: string,
  handlers: StreamHandlers,
): { unlisten: UnlistenFn } {
  const unlisten = listen<MessageStreamEvent>(agentStreamChannel(sessionId), (payload) => {
    if (!payload || typeof payload.type !== 'string') return
    switch (payload.type) {
      case 'stream_event':
        handlers.onStreamEvent?.(payload.data as StreamEventPayload)
        break
      case 'tool_call_start':
      case 'tool_call_complete':
        handlers.onToolEvent?.(payload)
        break
      case 'turn_start':
      case 'generation_start':
      case 'turn_complete':
      case 'turn_error':
      case 'compaction':
      default:
        handlers.onTurnEvent?.(payload)
        break
    }
  })
  return {
    unlisten: () => {
      unlisten()
    },
  }
}

/** Re-exported invoke helper for gateway commands not yet wrapped above. */
export { invokeCommand }
