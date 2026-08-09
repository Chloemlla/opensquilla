/**
 * Tauri event listener manager.
 *
 * In the WebSocket world the frontend subscribed to gateway events through
 * `rpc.on('session.event.text_delta', handler)` (see `src/lib/rpc.ts`). Under
 * Tauri the same events are emitted from Rust via `app_handle.emit(event, payload)`
 * and consumed in JS with `listen(event, handler)` from `@tauri-apps/api/event`.
 *
 * The Rust runtime emits these channels:
 *
 *   - `agent:stream:{sessionId}` — per-session turn/stream/tool events. The
 *     payload is `MessageStreamEvent` (see `src-tauri/src/ipc.rs`), a serde
 *     internally-tagged enum (`#[serde(tag="type", content="data")]`) whose
 *     `type` discriminator is one of `turn_start | generation_start |
 *     stream_event | tool_call_start | tool_call_complete | turn_complete |
 *     turn_error | compaction`. The `stream_event` data is itself an
 *     internally-tagged enum (`#[serde(tag="event", content="data")]`) whose
 *     `event` discriminator is `content_block_start | content_block_delta |
 *     content_block_stop | message_delta | message_stop | error | ping`.
 *     Field names inside `data` are snake_case (these enums are NOT
 *     `rename_all = "camelCase"`), while nested shared structs that are
 *     camelCase-renamed (e.g. `UsagePayload`) keep their camelCase fields.
 *   - `sessions:list-changed` — session list refresh (payload null).
 *   - `gateway:status` — gateway lifecycle changes (`{ running, url?, port?, error? }`).
 *   - `locale://changed` — locale override persisted (`{ locale }`).
 *   - `update://state` — updater state snapshot.
 *
 * This module provides:
 *  - {@link TauriEventListener} / {@link UnlistenFn}: transport-agnostic types.
 *  - {@link listen}: a low-level listener that talks to `window.__TAURI__.event`
 *    when present and no-ops (returning a NOP unlisten) when not, so dev mode
 *    without Tauri never crashes.
 *  - {@link useTauriEvent}: a Vue composable that binds a listener to the
 *    component lifecycle and auto-cleans up on unmount, mirroring
 *    `useRpcEvent` in `src/composables/useRpc.ts`.
 *  - {@link useTauriStreamEvents}: a Vue composable that subscribes to the
 *    per-session `agent:stream:{sessionId}` channel and dispatches each
 *    `MessageStreamEvent` to the turn / stream / tool handler groups by its
 *    `type` tag.
 */

import { onUnmounted } from 'vue'
import { isTauri, TauriInvokeError } from './invoke'

/** A function returned by `listen` that detaches the handler. */
export type UnlistenFn = () => void

/** A handler invoked for every matching Tauri event. */
export type TauriEventListener<T = unknown> = (payload: T) => void

/** Shape of the Tauri event API injected by the shell (@tauri-apps/api v2). */
export interface TauriEventApi {
  listen<T = unknown>(
    event: string,
    handler: (e: { event: string; id: number; payload: T }) => void,
  ): Promise<UnlistenFn>
}

export interface TauriEventGlobal {
  __TAURI__?: {
    event?: TauriEventApi
    core?: unknown
  }
}

/* ──────────────────────────────────────────────────────────────────────────
 * Wire payload types. These mirror the serde enums/structs in
 * `src-tauri/src/ipc.rs` exactly. `MessageStreamEvent` and `StreamEventPayload`
 * are internally tagged (`type` / `event`) with `data` content; their fields
 * are snake_case because those enums have no `rename_all`.
 * ────────────────────────────────────────────────────────────────────────── */

/** `ContentBlockDto` — a single content block inside a `MessageDto`. */
export interface ContentBlockDto {
  type: 'text' | 'tool_use' | 'tool_result' | 'reasoning' | (string & {})
  /** text blocks */
  text?: string
  /** tool_use blocks */
  id?: string
  name?: string
  input?: unknown
  /** tool_result blocks */
  tool_use_id?: string
  content?: string
  is_error?: boolean
  /** reasoning blocks */
  reasoning?: string
}

/** `MessageDto` — a message in a chat history or turn_complete payload. */
export interface MessageDto {
  role: string
  content: ContentBlockDto[]
  name?: string
  tool_call_id?: string
}

/** `UsagePayload` — camelCase-renamed token usage. */
export interface UsagePayload {
  inputTokens: number
  outputTokens: number
  totalTokens: number
}

/** `StreamDeltaPayload` — internally tagged stream delta. */
export type StreamDeltaPayload =
  | { type: 'text_delta'; text: string }
  | { type: 'reasoning_delta'; reasoning: string }
  | { type: 'input_json_delta'; partial_json: string }

/**
 * `StreamEventPayload` — the inner payload carried by a `stream_event`
 * `MessageStreamEvent`. Internally tagged with `event` and `data`.
 */
export type StreamEventPayload =
  | { event: 'content_block_start'; data: { index: number; block: ContentBlockDto } }
  | { event: 'content_block_delta'; data: { index: number; delta: StreamDeltaPayload } }
  | { event: 'content_block_stop'; data: { index: number } }
  | {
      event: 'message_delta'
      data: { stop_reason?: string | null; usage?: UsagePayload | null }
    }
  | { event: 'message_stop'; data: { content: ContentBlockDto[]; usage?: UsagePayload | null } }
  | { event: 'error'; data: { message: string; code?: string | null } }
  | { event: 'ping'; data?: null }

/**
 * `MessageStreamEvent` — the payload emitted on `agent:stream:{sessionId}`.
 * Internally tagged with `type` and `data`.
 */
export type MessageStreamEvent =
  | { type: 'turn_start'; data: { turn_id: string; session_id: string; timestamp: string } }
  | { type: 'generation_start'; data: { model: string; provider: string } }
  | { type: 'stream_event'; data: StreamEventPayload }
  | {
      type: 'tool_call_start'
      data: { tool_call_id: string; name: string; input: unknown }
    }
  | {
      type: 'tool_call_complete'
      data: {
        tool_call_id: string
        name: string
        content: string
        is_error: boolean
        duration_ms: number
      }
    }
  | {
      type: 'turn_complete'
      data: {
        turn_id: string
        session_id: string
        messages: MessageDto[]
        usage: UsagePayload
        duration_ms: number
      }
    }
  | {
      type: 'turn_error'
      data: { turn_id: string; session_id: string; message: string; code?: string | null }
    }
  | { type: 'compaction'; data: { before_count: number; after_count: number; success: boolean } }

/** The full channel name for a session's streaming events. */
export function agentStreamChannel(sessionId: string): string {
  return `agent:stream:${sessionId}`
}

/** `GatewayStatusEvent` — payload of the `gateway:status` channel. */
export interface GatewayStatusEvent {
  running: boolean
  url?: string | null
  port?: number | null
  error?: string | null
}

/** Payload of the `locale://changed` channel. */
export interface LocaleChangedEvent {
  locale: string
}

/** Payload of the `update://state` channel. */
export interface UpdateStateEvent {
  status: string
  available?: boolean
  version?: string | null
  release_url?: string | null
  error?: string | null
  downloading?: boolean
  downloaded?: boolean
  applying?: boolean
  [key: string]: unknown
}

/**
 * Mapping of the top-level Tauri event channel names to their payload types.
 * The streaming channel `agent:stream:{sessionId}` is session-scoped, so it is
 * not listed here (use {@link agentStreamChannel} + {@link listen} or
 * {@link useTauriStreamEvents}).
 */
export interface TauriEventMap {
  'sessions:list-changed': null
  'gateway:status': GatewayStatusEvent
  'locale://changed': LocaleChangedEvent
  'update://state': UpdateStateEvent
}

/** Union of all top-level Tauri event names. */
export type TauriEventName = keyof TauriEventMap

/** A Tauri event name OR an arbitrary string (for one-off listeners). */
export type EventName = TauriEventName | (string & {})

function getTauriEventApi(): TauriEventApi | null {
  const tauri = (globalThis as unknown as TauriEventGlobal).__TAURI__
  return tauri?.event ?? null
}

const nopUnlisten: UnlistenFn = () => {}

/**
 * Low-level Tauri event listener. Subscribes to `eventName` and invokes
 * `handler` for every matching emission. Returns an unlisten function.
 *
 * When not running in a Tauri shell (dev server, vitest), this returns a NOP
 * unlistener instead of throwing, so components and adapters can call it
 * unconditionally. Callers that need to know whether the subscription is live
 * should check {@link isTauri} first.
 */
export function listen<T = unknown>(
  eventName: EventName,
  handler: TauriEventListener<T>,
): UnlistenFn {
  const api = getTauriEventApi()
  if (!api) {
    return nopUnlisten
  }
  let active = true
  let detach: UnlistenFn | null = null
  // `listen` returns a Promise<UnlistenFn>; we capture it so a fast unmount
  // before resolution can still tear down once it lands.
  api.listen<T>(eventName, (e) => {
    if (!active) return
    handler(e.payload)
  })
    .then((un) => {
      if (!active) {
        // Component already unmounted before the listener attached.
        try { un() } catch { /* ignore */ }
        return
      }
      detach = un
    })
    .catch(() => {
      // A failed listen (e.g. event plugin missing on older shells) is
      // non-fatal; the caller simply won't receive events.
    })
  return () => {
    if (!active) return
    active = false
    if (detach) {
      try { detach() } catch { /* ignore */ }
      detach = null
    }
  }
}

/**
 * Vue composable: subscribe to a Tauri event for the lifetime of the calling
 * component and auto-cleanup on unmount. Mirrors `useRpcEvent` from
 * `src/composables/useRpc.ts`, so chat composables can swap transports with a
 * one-line change.
 *
 * Must be called from `setup()` (relies on `onUnmounted`).
 *
 * @example
 *   useTauriEvent('gateway:status', (p: GatewayStatusEvent) => {
 *     if (p.running) setGatewayUrl(p.url)
 *   })
 */
export function useTauriEvent<E extends TauriEventName>(
  eventName: E,
  handler: TauriEventListener<TauriEventMap[E]>,
): { unlisten: UnlistenFn }

export function useTauriEvent<T = unknown>(
  eventName: EventName,
  handler: TauriEventListener<T>,
): { unlisten: UnlistenFn }

export function useTauriEvent<T = unknown>(
  eventName: EventName,
  handler: TauriEventListener<T>,
): { unlisten: UnlistenFn } {
  const unlisten = listen<T>(eventName, handler)
  onUnmounted(() => {
    unlisten()
  })
  return { unlisten }
}

/**
 * Handler groups for the per-session streaming channel. Each `MessageStreamEvent`
 * is routed by its `type` tag:
 *
 *   turn_start | generation_start | turn_complete | turn_error | compaction → onTurnEvent
 *   stream_event                                                          → onStreamEvent
 *   tool_call_start | tool_call_complete                                  → onToolEvent
 */
export interface StreamHandlers {
  onTurnEvent?: TauriEventListener<MessageStreamEvent>
  onStreamEvent?: TauriEventListener<StreamEventPayload>
  onToolEvent?: TauriEventListener<MessageStreamEvent>
}

/**
 * Subscribe to a session's `agent:stream:{sessionId}` channel and dispatch to
 * the turn / stream / tool handler groups by the `MessageStreamEvent.type` tag.
 * Returns a single unlisten that tears down the underlying Tauri listener.
 *
 * Must be called from `setup()` (relies on `onUnmounted`). The listen should be
 * attached BEFORE `send_message` is invoked so no early turn events are missed.
 */
export function useTauriStreamEvents(
  sessionId: string,
  handlers: StreamHandlers,
): { unlisten: UnlistenFn } {
  const unlisten = listen<MessageStreamEvent>(agentStreamChannel(sessionId), (payload) => {
    if (!payload || typeof payload.type !== 'string') return
    switch (payload.type) {
      case 'stream_event':
        handlers.onStreamEvent?.(payload.data)
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
  onUnmounted(unlisten)
  return { unlisten }
}

/** Re-exported so adapters can pattern-match errors from listen setup. */
export { isTauri, TauriInvokeError }
