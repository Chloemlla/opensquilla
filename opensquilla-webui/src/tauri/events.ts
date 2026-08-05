/**
 * Tauri event listener manager.
 *
 * In the WebSocket world the frontend subscribed to gateway events through
 * `rpc.on('session.event.text_delta', handler)` (see `src/lib/rpc.ts`). Under
 * Tauri the same events are emitted from Rust via `app_handle.emit(event, payload)`
 * and consumed in JS with `listen(event, handler)` from `@tauri-apps/api/event`.
 *
 * This module provides:
 *  - {@link TauriEventListener} / {@link UnlistenFn}: transport-agnostic types.
 *  - {@link listen}: a low-level listener that talks to `window.__TAURI__.event`
 *    when present and no-ops (returning a NOP unlisten) when not, so dev mode
 *    without Tauri never crashes.
 *  - {@link useTauriEvent}: a Vue composable that binds a listener to the
 *    component lifecycle and auto-cleans up on unmount, mirroring
 *    `useRpcEvent` in `src/composables/useRpc.ts`.
 *  - {@link TauriEventMap}: event-name → payload type map mirroring the Rust
 *    IPC event enums. The streaming event names (`turn-event`, `stream-event`,
 *    `tool-event`) are the three top-level Tauri event channels the Rust
 *    runtime emits; their payloads are the discriminated unions below, which
 *    subsume the granular `session.event.*` names from `RpcEventMap`.
 *
 * The payload interfaces deliberately mirror the Rust serde structs
 * (`TurnEvent`, `StreamEvent`, `ToolEvent`) documented in
 * `docs/tauri-migration-analysis.md` P1.
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
 * Event payload types. These mirror the Rust IPC enums emitted by the runtime.
 * The three top-level channels consolidate the granular session.event.* names
 * used over WebSocket into broader discriminated unions, because the Rust
 * runtime groups them by transport concern (turn lifecycle / token stream /
 * tool execution). Each payload carries the original `event` discriminator so
 * the renderer can branch exactly as it did with RpcEventMap.
 * ────────────────────────────────────────────────────────────────────────── */

/** Common envelope embedded in every event payload. Mirrors Rust `EventEnvelope`. */
export interface TauriEventEnvelope {
  /** Session key this event belongs to (snake_case mirror of Rust field). */
  session_key?: string
  sessionKey?: string
  /** Epoch counter, bumped on compaction. */
  epoch?: number
  /** Monotonic per-stream sequence number. */
  stream_seq?: number
  /** Task id of the running turn. */
  task_id?: string
  taskId?: string
  /** Turn id of the running turn. */
  turn_id?: string
  turnId?: string
  [key: string]: unknown
}

/**
 * `turn-event` — turn-lifecycle events (state changes, compaction, heartbeat,
 * warnings, meta-skill progress, task-group transitions). Mirrors Rust
 * `TurnEvent` enum; the `event` field carries the original discriminator
 * (`session.event.state_change`, `session.event.compaction`, etc.) so the
 * renderer can reuse its existing per-event handlers.
 */
export interface TurnEventPayload extends TauriEventEnvelope {
  /** Original discriminator, e.g. `session.event.state_change`. */
  event: string
  /** Terminal/intermediate status (e.g. `running`, `completed`, `failed`). */
  status?: string
  /** Run status alias some gateway events use. */
  run_status?: string
  runStatus?: string
  /** Human-readable reason for a terminal transition. */
  reason?: string
  /** Terminal message text. */
  terminal_message?: string
  terminalMessage?: string
  /** Machine-readable terminal reason code. */
  terminal_reason?: string
  terminalReason?: string
  /** Warning/error code. */
  code?: string
  /** Free-form message. */
  message?: string
  /** Compaction status (only for compaction events). */
  compaction_status?: string
  compactionStatus?: string
  /** Active task descriptor. */
  active_task?: unknown
  activeTask?: unknown
  /** Last task descriptor. */
  last_task?: unknown
  lastTask?: unknown
  /** Target turn id for input disposition events. */
  target_turn_id?: string
  targetTurnId?: string
  /** Client request id correlation. */
  client_request_id?: string
  clientRequestId?: string
  /** Client message id correlation. */
  client_message_id?: string
  clientMessageId?: string
  payload?: unknown
}

/**
 * `stream-event` — token-stream events (text deltas, reasoning, artifacts,
 * router decisions, ensemble progress). Mirrors Rust `StreamEvent` enum.
 */
export interface StreamEventPayload extends TauriEventEnvelope {
  /** Original discriminator, e.g. `session.event.text_delta`. */
  event: string
  /** Incremental text fragment (text_delta). */
  text?: string
  /** Gateway-owned semantic role for a text span. */
  presentation?: 'intermediate' | 'answer'
  /** Artifact payload (artifact events). */
  artifact?: unknown
  /** Router decision payload (router_decision events). */
  router_decision?: unknown
  routerDecision?: unknown
  /** Ensemble progress payload (ensemble_progress events). */
  ensemble_progress?: unknown
  ensembleProgress?: unknown
  /** Routed model label. */
  model?: string
  /** Routing tier. */
  tier?: string
  payload?: unknown
}

/**
 * `tool-event` — tool-call lifecycle events (start, delta, result). Mirrors
 * Rust `ToolEvent` enum.
 */
export interface ToolEventPayload extends TauriEventEnvelope {
  /** Original discriminator: `session.event.tool_use_start|delta|result`. */
  event: string
  /** Stable tool-use id. */
  id?: string
  tool_use_id?: string
  toolUseId?: string
  /** Tool name. */
  name?: string
  tool_name?: string
  toolName?: string
  /** Parsed tool input (start/result). */
  input?: unknown
  /** Incremental input fragment (delta). */
  input_delta?: string
  inputDelta?: string
  /** Raw JSON fragment (delta). */
  json_fragment?: string
  jsonFragment?: string
  /** Alias some backends use for input_delta. */
  fragment?: string
  /** Tool result payload (result events). */
  result?: unknown
  content?: unknown
  output?: unknown
  error?: unknown
  is_error?: boolean
  isError?: boolean
  /** Server wall-clock tool start time (epoch ms). */
  started_at?: number
  /** Execution status object. */
  execution_status?: { status?: string }
  executionStatus?: { status?: string }
}

/** Other top-level Tauri event channels emitted outside a turn. */
export interface SessionsChangedPayload extends TauriEventEnvelope {
  event: 'sessions.changed'
}

export interface TaskQueuePayload extends TauriEventEnvelope {
  event: 'task.queued'
}

export interface TaskRunningPayload extends TauriEventEnvelope {
  event: 'task.running'
}

export interface EpochChangedPayload extends TauriEventEnvelope {
  event: 'session.epoch_changed'
}

export interface CronRunFinishedPayload extends TauriEventEnvelope {
  event: 'cron.run.finished'
}

/**
 * Mapping of the top-level Tauri event names to their payload types. The three
 * streaming channels (`turn-event`, `stream-event`, `tool-event`) are the
 * primary transport for chat updates; the remainder are session/task lifecycle
 * notifications emitted directly by the runtime.
 */
export interface TauriEventMap {
  'turn-event': TurnEventPayload
  'stream-event': StreamEventPayload
  'tool-event': ToolEventPayload
  'sessions.changed': SessionsChangedPayload
  'task.queued': TaskQueuePayload
  'task.running': TaskRunningPayload
  'session.epoch_changed': EpochChangedPayload
  'cron.run.finished': CronRunFinishedPayload
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
 *   useTauriEvent('stream-event', (p: StreamEventPayload) => {
 *     if (p.event === 'session.event.text_delta') appendText(p.text)
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
 * Subscribe to all three streaming channels at once and dispatch to typed
 * handlers. Returns a single unlisten that tears down all three. Used by the
 * gateway adapter's session subscription to mirror the granular
 * `rpc.on('session.event.*')` registrations in
 * `src/composables/chat/useChatRpcSubscriptions.ts`.
 */
export interface StreamHandlers {
  onTurnEvent?: TauriEventListener<TurnEventPayload>
  onStreamEvent?: TauriEventListener<StreamEventPayload>
  onToolEvent?: TauriEventListener<ToolEventPayload>
}

export function useTauriStreamEvents(handlers: StreamHandlers): { unlisten: UnlistenFn } {
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
  const unlisten: UnlistenFn = () => {
    for (const u of unlisteners) u()
    unlisteners.length = 0
  }
  onUnmounted(unlisten)
  return { unlisten }
}

/** Re-exported so adapters can pattern-match errors from listen setup. */
export { isTauri, TauriInvokeError }
