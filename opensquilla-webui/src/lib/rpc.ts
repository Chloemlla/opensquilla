/**
 * OpenSquilla Web UI — WebSocket RPC client (TypeScript port).
 *
 * This module owns the low-level RPC transports. The WebSocket {@link RpcClient}
 * (the original browser transport) is kept as the dev-mode fallback, while
 * {@link TauriRpcClient} provides a drop-in replacement that routes the same
 * `call()` / `on()` surface through the in-process Tauri bridge
 * (`src/tauri/invoke.ts` + `src/tauri/events.ts`) when the app runs inside the
 * Tauri shell. The Pinia RPC store (`src/stores/rpc.ts`) picks one of the two
 * based on {@link isTauri}.
 */

import { invoke, TauriInvokeError } from '@/tauri/invoke'
import {
  agentStreamChannel,
  listen,
  type MessageStreamEvent,
  type UnlistenFn,
} from '@/tauri/events'

export interface RpcErrorDetail {
  code?: string;
  message?: string;
  details?: unknown;
  retryable?: boolean;
  retry_after_ms?: number;
  accepted?: boolean;
}

export interface RpcClientError extends Error {
  code?: string;
  details?: unknown;
  retryable?: boolean;
  retry_after_ms?: number;
  accepted?: boolean;
}

export type RpcTerminationAction = 'reject' | 'reconnect';

export interface RpcCallOptions {
  timeoutMs?: number;
  signal?: AbortSignal;
  timeoutAction?: RpcTerminationAction;
  abortAction?: RpcTerminationAction;
  /** Called synchronously only after the request frame is accepted by send(). */
  onSent?: (socketGeneration: number) => void;
}

export interface RpcConnectionWaitOptions {
  timeoutAction?: RpcTerminationAction;
  abortAction?: RpcTerminationAction;
}

export class RpcTimeoutError extends Error implements RpcClientError {
  readonly code = 'RPC_TIMEOUT';

  constructor(
    readonly method: string,
    readonly timeoutMs: number
  ) {
    super(`${method} timed out after ${timeoutMs}ms`);
    this.name = 'RpcTimeoutError';
  }
}

export class RpcAbortError extends Error implements RpcClientError {
  readonly code = 'RPC_ABORTED';

  constructor(readonly method: string) {
    super(`${method} was aborted`);
    this.name = 'RpcAbortError';
  }
}

export interface RpcFrame {
  type?: string;
  id?: string;
  method?: string;
  params?: Record<string, unknown>;
  event?: string;
  payload?: unknown;
  meta?: Record<string, unknown>;
  ok?: boolean;
  error?: string | RpcErrorDetail;
  protocol?: number;
  policy?: Record<string, unknown>;
  features?: {
    methods?: string[];
    events?: string[];
  };
  auth?: Record<string, unknown>;
  seq?: number;
}

export type ConnectionState = 'disconnected' | 'connecting' | 'connected';
export type RpcEventHandler = {
  bivarianceHack(...args: unknown[]): void;
}['bivarianceHack'];

/**
 * The transport-agnostic surface both RPC clients implement. Consumers (the
 * Pinia RPC store, chat composables) depend on this shape, so swapping between
 * {@link RpcClient} (WebSocket) and {@link TauriRpcClient} (invoke/events) is a
 * constructor-level decision and nothing else changes.
 */
export interface RpcClientLike {
  /** Connect (WebSocket) or announce the bridge is ready (Tauri). */
  connect(url: string, token?: string): void;
  /** Disconnect / mark the transport offline. */
  disconnect(): void;
  /** Invoke a remote method and resolve with its result. */
  call(
    method: string,
    params?: Record<string, unknown>,
    options?: RpcCallOptions,
  ): Promise<unknown>;
  /** Subscribe to an event; returns an unsubscribe function. */
  on(event: string, handler: RpcEventHandler): () => void;
  /** Current connection state. */
  readonly state: ConnectionState;
  /** Gateway/hello policy object (may be empty). */
  readonly policy: Record<string, unknown>;
  /** Resolve once the transport is connected (or reject on timeout/abort). */
  waitForConnection(
    timeoutMs?: number,
    signal?: AbortSignal,
    actions?: RpcConnectionWaitOptions,
  ): Promise<void>;
}

interface PendingRequest {
  resolve: (value: unknown) => void;
  reject: (error: Error) => void;
  method: string;
  generation: number;
  timeoutTimer: ReturnType<typeof setTimeout> | null;
  signal: AbortSignal | null;
  abortHandler: (() => void) | null;
}

/** Max inbound WebSocket message size (32 MiB). Mirrors the Python gateway
 *  client `GATEWAY_CLIENT_MAX_MESSAGE_BYTES`; enforced in `onmessage` since
 *  the browser WS API has no configurable max_size. */
const GATEWAY_CLIENT_MAX_MESSAGE_BYTES = 32 * 1024 * 1024;

export class RpcClient implements RpcClientLike {
  private _ws: WebSocket | null = null;
  private _socketGeneration = 0;
  private _reqId = 0;
  private _pending = new Map<string, PendingRequest>();
  private _listeners = new Map<string, Set<RpcEventHandler>>();
  private _state: ConnectionState = 'disconnected';
  private _url = '';
  private _token: string | null = null;
  private _reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private _reconnectDelay = 800;
  private _maxReconnectDelay = 15000;
  private _reconnectFactor = 1.7;
  private _autoReconnect = true;
  private _pingTimer: ReturnType<typeof setInterval> | null = null;
  private _pingInterval = 55000;
  private _policy: Record<string, unknown> | null = null;
  private _lastSeq = 0;
  private _lastFrameAt = 0;
  private _tickWatchTimer: ReturnType<typeof setInterval> | null = null;
  private _tickTimeoutMs = 60000;

  connect(url: string, token?: string): void {
    this._url = url;
    this._token = token || null;
    this._autoReconnect = true;
    this._clearReconnectTimer();
    if (this._ws) {
      this._retireCurrentSocket(new Error('Connection replaced'), false);
    }
    this._doConnect();
  }

  disconnect(): void {
    this._autoReconnect = false;
    this._clearReconnectTimer();
    this._retireCurrentSocket(new Error('Disconnected'), false);
    this._rejectAllPending(new Error('Disconnected'));
    this._setState('disconnected');
  }

  call(
    method: string,
    params: Record<string, unknown> = {},
    options: RpcCallOptions = {}
  ): Promise<unknown> {
    return new Promise((resolve, reject) => {
      const socket = this._ws;
      const generation = this._socketGeneration;
      if (!socket || socket.readyState !== WebSocket.OPEN) {
        reject(new Error('Not connected'));
        return;
      }
      if (options.signal?.aborted) {
        reject(new RpcAbortError(method));
        return;
      }

      const id = String(++this._reqId);
      const pending: PendingRequest = {
        resolve,
        reject,
        method,
        generation,
        timeoutTimer: null,
        signal: options.signal || null,
        abortHandler: null,
      };
      this._pending.set(id, pending);

      const terminate = (error: Error, action: RpcTerminationAction): void => {
        if (!this._rejectPending(id, error, generation)) return;
        if (action === 'reconnect') {
          this._recycleConnection(
            generation,
            new Error(`Connection recycled after ${method} terminated`)
          );
        }
      };

      if (options.signal) {
        pending.abortHandler = () => {
          terminate(new RpcAbortError(method), options.abortAction || 'reject');
        };
        options.signal.addEventListener('abort', pending.abortHandler, { once: true });
      }

      if (
        options.timeoutMs !== undefined &&
        options.timeoutMs > 0 &&
        Number.isFinite(options.timeoutMs)
      ) {
        pending.timeoutTimer = setTimeout(() => {
          terminate(
            new RpcTimeoutError(method, options.timeoutMs!),
            options.timeoutAction || 'reject'
          );
        }, options.timeoutMs);
      }

      let frame: string;
      try {
        frame = JSON.stringify({ type: 'req', id, method, params });
      } catch (error) {
        this._rejectPending(
          id,
          error instanceof Error ? error : new Error('Failed to serialize RPC request'),
          generation
        );
        return;
      }

      try {
        socket.send(frame);
      } catch (error) {
        const sendError =
          error instanceof Error ? error : new Error('Failed to send RPC request');
        this._rejectPending(id, sendError, generation);
        this._recycleConnection(generation, sendError);
        return;
      }
      try {
        options.onSent?.(generation);
      } catch {
        // A send receipt is observational. It must never fail a request whose
        // frame is already on the wire.
      }
    });
  }

  on(event: string, handler: RpcEventHandler): () => void {
    if (!this._listeners.has(event)) this._listeners.set(event, new Set());
    this._listeners.get(event)!.add(handler);
    return () => this._listeners.get(event)?.delete(handler);
  }

  get state(): ConnectionState {
    return this._state;
  }

  get policy(): Record<string, unknown> {
    return this._policy || {};
  }

  waitForConnection(
    timeoutMs: number = 30000,
    signal?: AbortSignal,
    actions: RpcConnectionWaitOptions = {}
  ): Promise<void> {
    if (signal?.aborted) {
      // No wait and no request ever started, so this caller owns no socket to
      // recycle. Retiring the current connection here could kill a newer
      // session's healthy generation.
      return Promise.reject(new RpcAbortError('waitForConnection'));
    }
    if (this._state === 'connected') return Promise.resolve();

    return new Promise((resolve, reject) => {
      let timer: ReturnType<typeof setTimeout> | null = null;
      let settled = false;
      let off: () => void = () => {};

      const cleanup = (): void => {
        if (timer !== null) {
          clearTimeout(timer);
          timer = null;
        }
        off();
        signal?.removeEventListener('abort', onAbort);
      };
      const finish = (
        error?: Error,
        action: RpcTerminationAction = 'reject'
      ): void => {
        if (settled) return;
        settled = true;
        cleanup();
        if (!error) {
          resolve();
          return;
        }
        reject(error);
        if (action === 'reconnect') {
          // A disconnected waiter spans the reconnect gap. Its originally
          // observed generation may already have been retired before the
          // replacement handshake started; recycle the current still-
          // unconnected generation so the next attempt gets a fresh socket.
          if (this._state !== 'connected') {
            this._recycleConnection(
              this._socketGeneration,
              new Error('Connection recycled after waitForConnection terminated')
            );
          }
        }
      };
      const onAbort = (): void => {
        finish(
          new RpcAbortError('waitForConnection'),
          actions.abortAction || 'reject'
        );
      };

      off = this.on('_state', (s: ConnectionState) => {
        if (s === 'connected') {
          finish();
        }
      });
      signal?.addEventListener('abort', onAbort, { once: true });
      if (timeoutMs > 0 && Number.isFinite(timeoutMs)) {
        timer = setTimeout(() => {
          finish(
            new RpcTimeoutError('waitForConnection', timeoutMs),
            actions.timeoutAction || 'reject'
          );
        }, timeoutMs);
      }
    });
  }

  private _doConnect(): void {
    if (this._ws) return;
    this._setState('connecting');
    this._lastSeq = 0;
    this._lastFrameAt = Date.now();
    this._stopTickWatch();
    const generation = ++this._socketGeneration;
    let socket: WebSocket;
    try {
      socket = new WebSocket(this._url);
    } catch {
      if (generation !== this._socketGeneration) return;
      this._setState('disconnected');
      this._scheduleReconnect();
      return;
    }
    this._ws = socket;
    let handshakeRequestId: string | null = null;

    socket.onopen = () => {
      if (!this._isCurrentSocket(socket, generation)) return;
      this._reconnectDelay = 800;
      // Don't send connect yet — wait for connect.challenge from server
    };

    socket.onmessage = (ev: MessageEvent) => {
      if (!this._isCurrentSocket(socket, generation)) return;
      // UTF-16 code-unit cap approximating the 32 MiB byte cap. Browser WS has
      // no configurable max_size or asyncio-style receive queue (max_queue=1
      // has no direct port — messages arrive one at a time on the event loop).
      if (typeof ev.data === 'string' && ev.data.length > GATEWAY_CLIENT_MAX_MESSAGE_BYTES) {
        console.warn(
          `[rpc] inbound WebSocket frame exceeds ${GATEWAY_CLIENT_MAX_MESSAGE_BYTES} bytes; closing (1009)`
        );
        try {
          socket.close(1009, 'message too big');
        } catch {
          /* socket already closing */
        }
        this._recycleConnection(generation, new Error('Inbound message exceeds size cap'));
        return;
      }
      let data: RpcFrame;
      try {
        data = JSON.parse(ev.data);
      } catch {
        return;
      }
      if (!this._noteIncomingFrame(data)) return;

      // Handshake: server sends connect.challenge, we reply with connect request
      if (data.type === 'event' && data.event === 'connect.challenge') {
        const authParams = this._token ? { auth: { token: this._token } } : {};
        const id = String(++this._reqId);
        if (handshakeRequestId) return;
        handshakeRequestId = id;
        this._pending.set(id, {
          resolve: () => {},
          reject: (_err: Error) => {
            this._recycleConnection(generation, new Error('Connect handshake failed'));
          },
          method: 'connect',
          generation,
          timeoutTimer: null,
          signal: null,
          abortHandler: null,
        });
        try {
          socket.send(
            JSON.stringify({
              type: 'req',
              id,
              method: 'connect',
              params: {
                minProtocol: 3,
                maxProtocol: 3,
                client: { name: 'opensquilla-web' },
                ...authParams,
              },
            })
          );
        } catch (error) {
          const sendError =
            error instanceof Error ? error : new Error('Failed to send connect request');
          this._rejectPending(id, sendError, generation);
          this._recycleConnection(generation, sendError);
        }
        return;
      }

      // Handshake: HelloOk frame
      if (data.protocol !== undefined && this._state === 'connecting') {
        this._policy = data.policy || null;
        if (handshakeRequestId) {
          this._resolvePending(handshakeRequestId, data, generation);
          handshakeRequestId = null;
        }
        this._setState('connected');
        const helloHandlers = this._listeners.get('_hello');
        if (helloHandlers) helloHandlers.forEach((h) => h(data));
        this._startPing();
        this._startTickWatch();
        return;
      }

      if (data.type === 'res') {
        const id = data.id ?? '';
        if (data.ok) {
          this._resolvePending(id, data.payload, generation);
        } else {
          const err = data.error;
          const message =
            typeof err === 'string'
              ? err
              : (err && (err.message || err.code)) || 'RPC error';
          const error = new Error(message) as RpcClientError;
          if (err && typeof err === 'object') {
            error.code = err.code;
            error.details = err.details;
            error.retryable = err.retryable;
            error.retry_after_ms = err.retry_after_ms;
            error.accepted = err.accepted;
          }
          this._rejectPending(id, error, generation);
        }
      } else if (data.type === 'event') {
        const meta = data.meta || {};
        const handlers = this._listeners.get(data.event ?? '');
        if (handlers) handlers.forEach((h) => h(data.payload, meta));
        const wild = this._listeners.get('*');
        if (wild) wild.forEach((h) => h(data.event, data.payload, meta));
      }
    };

    socket.onclose = () => {
      if (!this._isCurrentSocket(socket, generation)) return;
      this._ws = null;
      ++this._socketGeneration;
      this._stopPing();
      this._stopTickWatch();
      this._rejectPendingForGeneration(generation, new Error('Connection closed'));
      this._setState('disconnected');
      this._scheduleReconnect();
    };

    socket.onerror = () => {};
  }

  private _isCurrentSocket(socket: WebSocket, generation: number): boolean {
    return this._ws === socket && this._socketGeneration === generation;
  }

  private _takePending(id: string, generation?: number): PendingRequest | undefined {
    const pending = this._pending.get(id);
    if (!pending || (generation !== undefined && pending.generation !== generation)) {
      return undefined;
    }
    this._pending.delete(id);
    if (pending.timeoutTimer !== null) {
      clearTimeout(pending.timeoutTimer);
      pending.timeoutTimer = null;
    }
    if (pending.signal && pending.abortHandler) {
      pending.signal.removeEventListener('abort', pending.abortHandler);
      pending.abortHandler = null;
    }
    return pending;
  }

  private _resolvePending(id: string, value: unknown, generation?: number): boolean {
    const pending = this._takePending(id, generation);
    if (!pending) return false;
    pending.resolve(value);
    return true;
  }

  private _rejectPending(id: string, error: Error, generation?: number): boolean {
    const pending = this._takePending(id, generation);
    if (!pending) return false;
    pending.reject(error);
    return true;
  }

  private _rejectPendingForGeneration(generation: number, error: Error): void {
    for (const [id, pending] of [...this._pending]) {
      if (pending.generation === generation) {
        this._rejectPending(id, error, generation);
      }
    }
  }

  private _rejectAllPending(error: Error): void {
    for (const id of [...this._pending.keys()]) {
      this._rejectPending(id, error);
    }
  }

  private _retireCurrentSocket(error: Error, reconnect: boolean): void {
    const socket = this._ws;
    const generation = this._socketGeneration;
    if (!socket) {
      this._stopPing();
      this._stopTickWatch();
      this._setState('disconnected');
      if (reconnect) this._scheduleReconnect(true);
      return;
    }

    this._ws = null;
    ++this._socketGeneration;
    this._stopPing();
    this._stopTickWatch();
    this._rejectPendingForGeneration(generation, error);
    this._setState('disconnected');
    try {
      socket.close();
    } catch {}
    if (reconnect) this._scheduleReconnect(true);
  }

  private _recycleConnection(generation: number, error: Error): void {
    if (generation !== this._socketGeneration) return;
    this._retireCurrentSocket(error, true);
  }

  private _clearReconnectTimer(): void {
    if (this._reconnectTimer !== null) {
      clearTimeout(this._reconnectTimer);
      this._reconnectTimer = null;
    }
  }

  private _startPing(): void {
    this._stopPing();
    this._pingTimer = setInterval(() => {
      if (this._ws && this._ws.readyState === WebSocket.OPEN) {
        this._ws.send('{"type":"ping"}');
      }
    }, this._pingInterval);
  }

  private _stopPing(): void {
    if (this._pingTimer !== null) {
      clearInterval(this._pingTimer);
      this._pingTimer = null;
    }
  }

  private _noteIncomingFrame(data: RpcFrame): boolean {
    this._lastFrameAt = Date.now();
    if (!data || data.type !== 'event' || typeof data.seq !== 'number') return true;

    const seq = data.seq;
    if (this._lastSeq > 0 && seq !== this._lastSeq + 1) {
      const detail = { expected: this._lastSeq + 1, actual: seq, event: data.event };
      const handlers = this._listeners.get('_gap');
      if (handlers) handlers.forEach((h) => h(detail));
      try {
        this._ws?.close();
      } catch {}
      return false;
    }
    this._lastSeq = seq;
    return true;
  }

  private _startTickWatch(): void {
    this._stopTickWatch();
    const tickMs = (this._policy?.tick_interval_ms as number) || 30000;
    this._tickTimeoutMs = Math.max(10000, tickMs * 2.5);
    this._lastFrameAt = Date.now();
    this._tickWatchTimer = setInterval(() => {
      if (!this._ws || this._ws.readyState !== WebSocket.OPEN) return;
      const idleMs = Date.now() - this._lastFrameAt;
      if (idleMs <= this._tickTimeoutMs) return;
      const handlers = this._listeners.get('_gap');
      if (handlers) handlers.forEach((h) => h({ reason: 'tick_timeout', idleMs }));
      try {
        this._ws.close();
      } catch {}
    }, Math.min(tickMs, 10000));
  }

  private _stopTickWatch(): void {
    if (this._tickWatchTimer !== null) {
      clearInterval(this._tickWatchTimer);
      this._tickWatchTimer = null;
    }
  }

  private _scheduleReconnect(immediate: boolean = false): void {
    if (!this._autoReconnect) return;
    this._clearReconnectTimer();
    const delay = immediate ? 0 : this._reconnectDelay;
    this._reconnectTimer = setTimeout(() => {
      this._reconnectTimer = null;
      if (!this._autoReconnect || this._ws) return;
      this._doConnect();
    }, delay);
    if (immediate) return;
    this._reconnectDelay = Math.min(
      this._reconnectDelay * this._reconnectFactor,
      this._maxReconnectDelay
    );
  }

  private _setState(s: ConnectionState): void {
    if (this._state === s) return;
    this._state = s;
    const handlers = this._listeners.get('_state');
    if (handlers) handlers.forEach((h) => h(s));
  }
}

/* ──────────────────────────────────────────────────────────────────────────
 * TauriRpcClient
 *
 * A drop-in replacement for {@link RpcClient} that runs entirely inside the
 * Tauri process. Requests are routed through `invoke()` (see
 * `src/tauri/invoke.ts`) and streaming events arrive on the per-session Tauri
 * channel `agent:stream:{sessionId}` (plus the top-level lifecycle channels
 * like `sessions:list-changed`), defined in `src/tauri/events.ts`.
 *
 * The WebSocket RPC gateway addressed methods by dotted names
 * (`chat.send`, `config.patch.safe`, …) and streamed granular events
 * (`session.event.text_delta`, …). Under Tauri, commands are snake_case and
 * events are routed by the `MessageStreamEvent.type` tag onto the granular
 * `session.event.*` names. {@link TAURI_METHOD_REGISTRY} is the living map
 * between the two worlds; extend it as the Rust command surface grows.
 *
 * The store creates this class only when {@link isTauri} is true, so the
 * WebSocket `RpcClient` above remains the browser/dev fallback.
 * ────────────────────────────────────────────────────────────────────────── */

/** Error thrown by {@link TauriRpcClient} for transport-level failures. */
export class TauriRpcError extends Error implements RpcClientError {
  readonly code?: string;
  readonly details?: unknown;
  readonly retryable?: boolean;
  readonly retry_after_ms?: number;
  readonly accepted?: boolean;

  constructor(code: string | undefined, message: string, details?: unknown) {
    super(message);
    this.name = 'TauriRpcError';
    this.code = code;
    this.details = details;
  }
}

/** Binds a dotted RPC method name to a Tauri command invocation. */
export interface TauriMethodBinding {
  /** The `tauri::command` name to invoke (used when `run` is absent). */
  command: string;
  /**
   * Maps the WebSocket-RPC params object onto the command's argument object.
   * Defaults to a shallow copy of the params.
   */
  transform?: (params: Record<string, unknown>) => Record<string, unknown>;
  /**
   * Custom executor for methods whose request/response shape diverges from a
   * 1:1 command call (e.g. batch `sessions.delete`). Takes precedence over
   * `command` + `transform`.
   */
  run?: (params: Record<string, unknown>) => Promise<unknown>;
}

/** Read the first present (non-null/undefined) param among `keys`. */
function firstParam(
  params: Record<string, unknown>,
  ...keys: string[]
): unknown {
  for (const key of keys) {
    const value = params[key];
    if (value !== undefined && value !== null) return value;
  }
  return undefined;
}

/**
 * Normalize the `config.patch` payload. The legacy gateway accepted either an
 * array of `{ path, value }` patches or a flat `{ 'dotted.path': value }` map
 * (the map form is what the locale sync + feature-toggle call sites use); the
 * Rust `patch_config` command takes the `{ key, value }` array form.
 */
function normalizeConfigPatches(
  raw: unknown,
): Array<{ key: string; value: unknown }> {
  if (Array.isArray(raw)) {
    return (raw as Array<{ key?: string; path?: string; value?: unknown }>).map((p) => ({
      key: p.key ?? p.path ?? '',
      value: p.value,
    }))
  }
  if (raw && typeof raw === 'object') {
    return Object.entries(raw as Record<string, unknown>).map(
      ([key, value]) => ({ key, value }),
    )
  }
  return []
}

/** Batch `sessions.delete` executor — mirrors the gateway's `keys: string[]` form. */
async function runSessionsDelete(
  params: Record<string, unknown>,
): Promise<unknown> {
  const rawKeys = params.keys ?? params.ids;
  const keys = Array.isArray(rawKeys)
    ? rawKeys.filter((key): key is string => typeof key === 'string')
    : [];
  if (keys.length > 0) {
    const results = await Promise.all(
      keys.map((id) => invoke<boolean>('delete_session', { sessionId: id })),
    );
    const deleted: string[] = [];
    const errors: string[] = [];
    keys.forEach((key, index) => {
      if (results[index] === true) {
        deleted.push(key);
      } else {
        errors.push('delete failed');
      }
    });
    return { deleted, errors };
  }
  const id = firstParam(params, 'id', 'key', 'sessionKey', 'session_key');
  if (typeof id === 'string' && id) {
    const ok = await invoke<boolean>('delete_session', { sessionId: id });
    return ok
      ? { deleted: [id], errors: [] }
      : { deleted: [], errors: ['delete failed'] };
  }
  return { deleted: [], errors: [] };
}

/**
 * `chat.send` executor. Attaches the per-session `agent:stream:{sessionId}`
 * listener BEFORE invoking `send_message` (the caller is expected to have
 * registered `session.event.*` handlers via `rpc.on` first), then wraps the
 * params into the `MessageSendRequest` envelope. Returns the turn id string.
 */
/**
 * Adapt a Rust `MessageDto` (role + typed content blocks) into the partial
 * `ChatHistoryMessage` shape the chat composables map. The Rust skeleton does
 * not yet persist turn_context/usage/timestamps, so only role, text and tool
 * calls survive the bridge; reconciliation falls back to role:ts:text keys.
 */
function adaptHistoryMessages(raw: unknown[]): Array<Record<string, unknown>> {
  return (raw ?? []).map((m) => {
    const msg = (m ?? {}) as Record<string, unknown>;
    const blocks = Array.isArray(msg.content)
      ? (msg.content as Array<Record<string, unknown>>)
      : [];
    const text = blocks
      .filter((b) => b.type === 'text')
      .map((b) => (typeof b.text === 'string' ? b.text : ''))
      .join('');
    const toolCalls = blocks
      .filter((b) => b.type === 'tool_use')
      .map((b) => ({ id: b.id, name: b.name, input: b.input }));
    return {
      role: msg.role,
      text,
      tool_calls: toolCalls,
    };
  });
}

async function runChatSend(
  params: Record<string, unknown>,
): Promise<unknown> {
  const sessionId = firstParam(params, 'sessionKey', 'session_key', 'sessionId', 'session_id');
  const request: Record<string, unknown> = {
    sessionId,
    message: params.message ?? params.content,
  };
  if (Array.isArray(params.history)) request.history = params.history;
  if (params.model !== undefined) request.model = params.model;
  if (params.provider !== undefined) request.provider = params.provider;
  if (params.stream !== undefined) request.stream = params.stream;
  const turnId = await invoke<string>('send_message', { request });
  // `send_message` resolves with the bare turn id; the chat composables expect
  // the gateway's ChatSendResponse envelope, so wrap it to keep the Tauri path
  // equivalent to the WebSocket path.
  return {
    sessionKey: sessionId,
    task_id: turnId,
    taskId: turnId,
    task_status: 'running',
    taskStatus: 'running',
  };
}

/**
 * Registry mapping the WebSocket RPC method names the WebUI calls onto the
 * Tauri commands exposed by the Rust runtime. Methods without an entry are
 * reported as `METHOD_NOT_FOUND` so consumers can mark them unavailable
 * (matching the gateway behavior when it lacks a method).
 */
export const TAURI_METHOD_REGISTRY: Record<string, TauriMethodBinding> = {
  // ── chat ────────────────────────────────────────────────────────────────
  'chat.send': {
    command: 'send_message',
    // runChatSend attaches the `agent:stream:{sessionId}` listener before
    // invoking `send_message` and wraps the request envelope.
    run: runChatSend,
  },
  'chat.abort': {
    command: 'abort_session',
    transform: (p) => ({ sessionId: firstParam(p, 'sessionKey', 'session_key', 'id', 'key') }),
  },
  'chat.cancel': {
    command: 'cancel_turn',
    transform: (p) => ({ sessionId: firstParam(p, 'sessionKey', 'session_key', 'sessionId', 'session_id', 'id') }),
  },
  'chat.history': {
    command: 'get_chat_history',
    transform: (p) => ({ sessionId: firstParam(p, 'sessionKey', 'session_key', 'key', 'id') }),
    // The Rust command returns a bare `MessageDto[]`; the composables expect
    // `{ messages }`, so wrap and adapt the shape.
    run: (p) =>
      invoke<unknown[]>('get_chat_history', {
        sessionId: firstParam(p, 'sessionKey', 'session_key', 'key', 'id'),
      }).then((r) => ({ messages: adaptHistoryMessages(r) })),
  },
  'chat.clear': {
    command: 'clear_chat_history',
    transform: (p) => ({ sessionId: firstParam(p, 'sessionKey', 'session_key', 'sessionId', 'session_id', 'id') }),
  },

  // ── sessions ────────────────────────────────────────────────────────────
  'sessions.list': {
    command: 'list_sessions',
    transform: () => ({}),
  },
  'sessions.create': {
    command: 'create_session',
    transform: (p) => ({
      request: {
        title: p.title,
        model: p.model,
        agentId: p.agentId ?? p.agent_id,
        systemPrompt: p.systemPrompt ?? p.system_prompt,
        mode: p.mode,
      },
    }),
  },
  'sessions.get': {
    command: 'get_session',
    transform: (p) => ({ sessionId: firstParam(p, 'id', 'key', 'sessionKey', 'session_key') }),
  },
  'sessions.preview': {
    // The inspect drawer calls with `{ keys: [key] }` and expects
    // `{ previews: [{ key, title, lastMessage, updatedAt }] }`. Rust
    // get_session returns `{ session: SessionInfo }` (camelCase), so map it.
    command: 'get_session',
    run: async (p) => {
      const keys = Array.isArray(p.keys) ? p.keys : p.key != null ? [p.key] : [];
      const key = String(keys[0] ?? '');
      if (!key) return { previews: [] };
      try {
        const res = await invoke<{
          session?: { id?: string; title?: string; updatedAt?: string };
        }>('get_session', { sessionId: key });
        const s = res?.session;
        return s?.id
          ? {
              previews: [
                {
                  key: s.id,
                  title: s.title ?? '',
                  lastMessage: '',
                  updatedAt: s.updatedAt ? Date.parse(s.updatedAt) || null : null,
                },
              ],
            }
          : { previews: [] };
      } catch {
        return { previews: [] };
      }
    },
  },
  'sessions.delete': {
    command: 'delete_session',
    transform: (p) => ({ sessionId: firstParam(p, 'id', 'key', 'sessionKey', 'session_key') }),
    run: runSessionsDelete,
  },
  'sessions.archive': {
    command: 'archive_session',
    transform: (p) => ({ sessionId: firstParam(p, 'id', 'key', 'sessionKey', 'session_key') }),
  },
  'sessions.fork': {
    command: 'fork_session',
    // `fork_session` resolves `{ session: SessionInfo }`; ChatView reads the
    // child session key via `res.key`, so unwrap to that shape.
    run: async (p) => {
      const args: Record<string, unknown> = {
        sessionId: firstParam(p, 'key', 'sessionKey', 'session_key', 'id'),
      };
      if (p.forkEvent !== undefined) args.forkEvent = p.forkEvent;
      if (p.title !== undefined) args.title = p.title;
      const res = await invoke<{ session?: { id?: string } }>('fork_session', args);
      return { key: res?.session?.id ?? '' };
    },
  },
  'sessions.compact': {
    command: 'compact_session',
    transform: (p) => ({ sessionId: firstParam(p, 'id', 'key', 'sessionKey', 'session_key') }),
  },
  'sessions.export': {
    command: 'export_session',
    transform: (p) => ({ sessionId: firstParam(p, 'sessionId', 'session_id', 'id') }),
  },
  'sessions.import': {
    command: 'import_session',
    transform: (p) => ({ payload: p.payload ?? p.session }),
  },

  // ── config ──────────────────────────────────────────────────────────────
  'config.get': {
    command: 'get_config',
    run: (_p) => invoke<{ config: unknown }>('get_config', {}).then((r) => r.config),
  },
  'config.effective': {
    command: 'get_config_effective',
    run: (_p) => invoke<{ config: unknown }>('get_config_effective', {}).then((r) => r.config),
  },
  'config.set': {
    command: 'set_config',
    transform: (p) => ({ request: { key: p.key, value: p.value } }),
  },
  'config.patch': {
    command: 'patch_config',
    transform: (p) => ({
      patches: normalizeConfigPatches(p.patches),
      safe: p.safe ?? false,
    }),
  },
  'config.patch.safe': {
    command: 'patch_config',
    transform: (p) => ({
      patches: normalizeConfigPatches(p.patches),
      safe: true,
    }),
  },
  'config.list': { command: 'list_config' },
  'config.reset': {
    command: 'reset_config',
    transform: (p) => (p.key !== undefined ? { key: p.key } : {}),
  },
  'config.value.get': {
    command: 'get_config_value',
    transform: (p) => ({ key: p.key }),
  },

  // ── providers / models ──────────────────────────────────────────────────
  'providers.list': {
    command: 'list_providers',
    run: (_p) => invoke<{ providers: unknown[] }>('list_providers', {}).then((r) => r.providers),
  },
  'providers.status': {
    // OverviewView reads `data.providers` as an array. Rust
    // get_all_provider_statuses returns `{ providers, defaultProvider, count }`.
    command: 'get_all_provider_statuses',
    run: (_p) => invoke<{ providers: unknown[] }>('get_all_provider_statuses', {}),
  },
  // models.routing.get is intentionally unbridged: the Rust runtime has no
  // routing-snapshot command, so callers degrade to the config projection
  // (METHOD_NOT_FOUND is caught in useChatFeatureToggles).
  'models.list': {
    command: 'list_models',
    run: (_p) => invoke<{ models: unknown[] }>('list_models', {}).then((r) => r.models),
  },

  // ── onboarding (S3 read-only rescue + S4 write methods) ──────────────────
  'onboarding.catalog': { command: 'onboarding_catalog' },
  'onboarding.status': { command: 'onboarding_status' },
  // S4 — provider configuration (single `request` struct arg)
  'onboarding.provider.configure': {
    command: 'onboarding_provider_configure',
    transform: (p) => ({ request: p }),
  },
  'onboarding.provider.credential.reveal': {
    command: 'onboarding_provider_credential_reveal',
    transform: (p) => ({ request: p }),
  },
  'onboarding.provider.credential.clear': {
    command: 'onboarding_provider_credential_clear',
    transform: (p) => ({ request: p }),
  },
  // S4 — LLM profiles
  'onboarding.llmProfile.upsert': {
    command: 'onboarding_llm_profile_upsert',
    transform: (p) => ({ request: p }),
  },
  'onboarding.llmProfile.credential.clear': {
    command: 'onboarding_llm_profile_credential_clear',
    transform: (p) => ({ request: p }),
  },
  'onboarding.llmProfile.remove': {
    command: 'onboarding_llm_profile_remove',
    transform: (p) => ({ request: p }),
  },
  'onboarding.llmProfile.activate': {
    command: 'onboarding_llm_profile_activate',
    transform: (p) => ({ request: p }),
  },
  'onboarding.llmProfile.active.remove': {
    command: 'onboarding_llm_profile_active_remove',
    transform: (p) => ({ request: p }),
  },
  'onboarding.capability.reset': {
    command: 'onboarding_capability_reset',
    transform: (p) => ({ request: p }),
  },
  // S4 — feature section config (flat params, Tauri auto-maps camelCase args)
  'onboarding.router.configure': {
    command: 'onboarding_router_configure',
    transform: (p) => p,
  },
  'onboarding.ensemble.configure': {
    command: 'onboarding_ensemble_configure',
    transform: (p) => p,
  },
  'onboarding.search.configure': {
    command: 'onboarding_search_configure',
    transform: (p) => p,
  },
  'onboarding.imageGeneration.configure': {
    command: 'onboarding_image_generation_configure',
    transform: (p) => p,
  },
  'onboarding.memory_embedding.configure': {
    command: 'onboarding_memory_embedding_configure',
    transform: (p) => p,
  },
  'onboarding.audio.configure': {
    command: 'onboarding_audio_configure',
    transform: (p) => p,
  },
  // S4 — channels editor (single `request` struct arg)
  'onboarding.channel.probe': {
    command: 'onboarding_channel_probe',
    transform: (p) => ({ request: p }),
  },
  'onboarding.channel.upsert': {
    command: 'onboarding_channel_upsert',
    transform: (p) => ({ request: p }),
  },
  'onboarding.channel.remove': {
    command: 'onboarding_channel_remove',
    transform: (p) => ({ request: p }),
  },
  'onboarding.channel.enable': {
    command: 'onboarding_channel_enable',
    transform: (p) => ({ request: p }),
  },
  'onboarding.channel.disable': {
    command: 'onboarding_channel_disable',
    transform: (p) => ({ request: p }),
  },

  // ── skills ──────────────────────────────────────────────────────────────
  'skills.list': { command: 'list_skills' },

  // ── system / desktop ────────────────────────────────────────────────────
  'system.health': { command: 'health_check' },
  doctor: { command: 'health_check' },
  'locale.get': { command: 'get_locale' },
  'locale.set': { command: 'set_locale', transform: (p) => ({ input: { locale: p.locale } }) },
  'updates.check': { command: 'check_updates' },
  'updates.install': { command: 'install_update' },
  'system.openExternal': {
    command: 'open_external',
    transform: (p) => ({ target: firstParam(p, 'url', 'target') }),
  },
  'system.pickDirectory': {
    command: 'pick_directory',
    transform: (p) => ({ initialPath: p.initialPath ?? p.initial_path }),
  },
  'zoom.in': { command: 'zoom_in' },
  'zoom.out': { command: 'zoom_out' },
  'zoom.reset': { command: 'zoom_reset' },
  'app.ping': { command: 'ping' },
  'app.info': { command: 'app_info' },

  // ── gateway lifecycle ───────────────────────────────────────────────────
  'gateway.start': { command: 'start_gateway' },
  'gateway.stop': { command: 'stop_gateway' },
  'gateway.status': { command: 'gateway_status' },
  'gateway.restart': { command: 'restart_gateway' },
  'gateway.url': { command: 'get_gateway_url' },
};

/**
 * The method surface the Tauri bridge advertises via the synthesized `_hello`
 * event. `supportsMethod()` in the RPC store derives from this, so only
 * methods with a command binding are reported available.
 */
export const TAURI_SUPPORTED_METHODS: readonly string[] = Object.freeze(
  Object.keys(TAURI_METHOD_REGISTRY),
);

/** Event names that are internal to the RPC client itself, not Tauri events. */
function isInternalEvent(event: string): boolean {
  return event === '_state' || event === '_hello' || event === '_gap';
}

/**
 * Events that arrive inside the per-session `agent:stream:{sessionId}` channel,
 * discriminated by the payload's `type` tag. Everything else is treated as a
 * top-level Tauri channel name and subscribed directly.
 */
function isStreamingDiscriminator(event: string): boolean {
  return event.startsWith('session.event.') || event === 'session.epoch_changed';
}

/** Top-level RPC lifecycle event names the wildcard `*` subscription attaches. */
const TAURI_TOP_LEVEL_EVENTS: readonly string[] = [
  'sessions.changed',
  'task.queued',
  'task.running',
  'cron.run.finished',
];

/**
 * Maps an RPC event name onto the actual Tauri channel the Rust runtime emits.
 * Entries that differ from the RPC name (the Rust side emits
 * `sessions:list-changed`, not `sessions.changed`) are listed here; everything
 * else subscribes to the same-named channel.
 */
const TAURI_EVENT_CHANNEL: Record<string, string> = {
  'sessions.changed': 'sessions:list-changed',
};

/**
 * Map a `MessageStreamEvent` (the `agent:stream:{sessionId}` payload) onto the
 * granular `session.event.*` RPC names the chat composables subscribe to.
 * `stream_event` payloads are further discriminated by their inner `event` tag.
 */
function mapMessageStreamEvent(
  payload: MessageStreamEvent,
): Array<{ name: string; data: unknown }> {
  if (!payload || typeof payload.type !== 'string') return [];
  switch (payload.type) {
    case 'turn_start':
      return [{
        name: 'session.event.state_change',
        data: {
          event: 'session.event.state_change',
          status: 'running',
          turn_id: payload.data.turn_id,
          session_id: payload.data.session_id,
          timestamp: payload.data.timestamp,
        },
      }];
    case 'generation_start':
      return [{
        name: 'session.event.state_change',
        data: {
          event: 'session.event.state_change',
          status: 'generating',
          model: payload.data.model,
          provider: payload.data.provider,
        },
      }];
    case 'stream_event': {
      const inner = payload.data;
      if (!inner || typeof inner.event !== 'string') return [];
      switch (inner.event) {
        case 'content_block_delta': {
          const delta = inner.data.delta;
          if (delta?.type === 'text_delta') {
            return [{
              name: 'session.event.text_delta',
              data: { event: 'session.event.text_delta', text: delta.text },
            }];
          }
          if (delta?.type === 'reasoning_delta') {
            return [{
              name: 'session.event.thinking',
              data: { event: 'session.event.thinking', reasoning: delta.reasoning },
            }];
          }
          if (delta?.type === 'input_json_delta') {
            return [{
              name: 'session.event.tool_use_delta',
              data: {
                event: 'session.event.tool_use_delta',
                json_fragment: delta.partial_json,
                jsonFragment: delta.partial_json,
              },
            }];
          }
          return [];
        }
        case 'content_block_start': {
          const block = inner.data.block;
          if (block?.type === 'tool_use') {
            return [{
              name: 'session.event.tool_use_start',
              data: {
                event: 'session.event.tool_use_start',
                id: block.id,
                tool_use_id: block.id,
                toolUseId: block.id,
                name: block.name,
                input: block.input,
              },
            }];
          }
          return [];
        }
        case 'error':
          return [{
            name: 'session.event.warning',
            data: {
              event: 'session.event.warning',
              message: inner.data.message,
              code: inner.data.code ?? 'STREAM_ERROR',
            },
          }];
        case 'message_delta':
        case 'content_block_stop':
        case 'message_stop':
        case 'ping':
        default:
          return [];
      }
    }
    case 'tool_call_start':
      return [{
        name: 'session.event.tool_use_start',
        data: {
          event: 'session.event.tool_use_start',
          id: payload.data.tool_call_id,
          tool_use_id: payload.data.tool_call_id,
          toolUseId: payload.data.tool_call_id,
          name: payload.data.name,
          input: payload.data.input,
        },
      }];
    case 'tool_call_complete':
      return [{
        name: 'session.event.tool_result',
        data: {
          event: 'session.event.tool_result',
          id: payload.data.tool_call_id,
          tool_use_id: payload.data.tool_call_id,
          toolUseId: payload.data.tool_call_id,
          name: payload.data.name,
          content: payload.data.content,
          is_error: payload.data.is_error,
          isError: payload.data.is_error,
        },
      }];
    case 'turn_complete':
      return [{
        name: 'session.event.state_change',
        data: {
          event: 'session.event.state_change',
          status: 'completed',
          turn_id: payload.data.turn_id,
          session_id: payload.data.session_id,
          messages: payload.data.messages,
          usage: payload.data.usage,
          duration_ms: payload.data.duration_ms,
        },
      }];
    case 'turn_error':
      return [
        {
          name: 'session.event.warning',
          data: {
            event: 'session.event.warning',
            message: payload.data.message,
            code: payload.data.code ?? 'TURN_ERROR',
          },
        },
        {
          name: 'session.event.state_change',
          data: {
            event: 'session.event.state_change',
            status: 'failed',
            turn_id: payload.data.turn_id,
            session_id: payload.data.session_id,
            message: payload.data.message,
            code: payload.data.code,
          },
        },
      ];
    case 'compaction':
      return [{
        name: 'session.event.compaction',
        data: {
          event: 'session.event.compaction',
          before_count: payload.data.before_count,
          after_count: payload.data.after_count,
          success: payload.data.success,
        },
      }];
    default:
      return [];
  }
}

/** Translate a Tauri command failure into an RpcClientError-compatible shape. */
function normalizeTauriError(method: string, err: unknown): Error {
  if (err instanceof TauriInvokeError) {
    if (
      err.code === 'COMMAND_NOT_FOUND'
      || err.code === 'METHOD_NOT_FOUND'
      || err.code === 'TAURI_UNAVAILABLE'
    ) {
      return new TauriRpcError(
        'METHOD_NOT_FOUND',
        `Method '${method}' is not registered in the Tauri bridge`,
        { method, cause: err },
      );
    }
    return err;
  }
  return err instanceof Error ? err : new TauriRpcError(undefined, String(err));
}

/** Apply timeout / abort / onSent semantics to a Tauri invoke promise. */
function applyCallOptions(
  promise: Promise<unknown>,
  method: string,
  options: RpcCallOptions,
): Promise<unknown> {
  return new Promise((resolve, reject) => {
    let settled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;

    const cleanup = (): void => {
      if (timer !== null) {
        clearTimeout(timer);
        timer = null;
      }
      options.signal?.removeEventListener('abort', onAbort);
    };
    const finish = (fn: () => void): void => {
      if (settled) return;
      settled = true;
      cleanup();
      fn();
    };
    const onAbort = (): void => {
      finish(() => reject(new RpcAbortError(method)));
    };

    if (options.signal?.aborted) {
      reject(new RpcAbortError(method));
      return;
    }
    options.signal?.addEventListener('abort', onAbort, { once: true });
    if (
      options.timeoutMs !== undefined
      && options.timeoutMs > 0
      && Number.isFinite(options.timeoutMs)
    ) {
      timer = setTimeout(() => {
        finish(() => reject(new RpcTimeoutError(method, options.timeoutMs as number)));
      }, options.timeoutMs);
    }
    try {
      options.onSent?.(0);
    } catch {
      // A send receipt is observational; it must never fail the request.
    }
    promise.then(
      (value) => finish(() => resolve(value)),
      (error) => finish(() => reject(error as Error)),
    );
  });
}

/**
 * Tauri-backed RPC client implementing the same surface as {@link RpcClient}.
 *
 * `call()` resolves the dotted RPC method name through
 * {@link TAURI_METHOD_REGISTRY} and runs the matching `invoke()` (or a custom
 * executor). `on()` maps RPC event subscriptions onto Tauri event listeners:
 * the granular `session.event.*` names are delivered via the per-session
 * `agent:stream:{sessionId}` channel, which `chat.send` attaches BEFORE
 * invoking `send_message` and routes by the payload's `type` tag; every other
 * name is subscribed directly on the same-named Tauri channel (via
 * {@link TAURI_EVENT_CHANNEL} when the RPC name differs from the channel).
 * `_state` / `_hello` / `_gap` are synthesized in-process so the Pinia store's
 * connection bookkeeping works unchanged.
 *
 * Construct this only when `isTauri()` is true.
 */
export class TauriRpcClient implements RpcClientLike {
  private _listeners = new Map<string, Set<RpcEventHandler>>();
  private _state: ConnectionState = 'disconnected';
  private _policy: Record<string, unknown> = {};
  private _directUnlisteners = new Map<string, UnlistenFn>();
  /**
   * One `agent:stream:{sessionId}` listener per session. Attached lazily by
   * `chat.send` so streaming events are captured before the turn starts.
   */
  private _sessionStreams = new Map<string, UnlistenFn>();
  /**
   * Reference count per direct Tauri listener. A single top-level channel (e.g.
   * `sessions:list-changed`) can be held by both a specific `on(name)` subscription
   * and the `*` wildcard, so the underlying listener must only be torn down
   * when the last holder releases it.
   */
  private _directRefs = new Map<string, number>();

  /**
   * Mark the in-process bridge ready. The `url`/`token` arguments are accepted
   * for {@link RpcClientLike} compatibility but are irrelevant under Tauri —
   * there is no socket to open and no gateway handshake to perform.
   */
  connect(_url: string, _token?: string): void {
    if (this._state === 'connected') return;
    this._setState('connecting');
    this._setState('connected');
    this._dispatch('_hello', {
      policy: {},
      auth: { principal: { isOwner: true } },
      features: { methods: TAURI_SUPPORTED_METHODS },
    });
  }

  /** Mark the transport offline. Tauri listeners remain attached. */
  disconnect(): void {
    this._setState('disconnected');
  }

  async call(
    method: string,
    params: Record<string, unknown> = {},
    options: RpcCallOptions = {},
  ): Promise<unknown> {
    if (this._state !== 'connected') {
      throw new Error(`Not connected (state: ${this._state})`);
    }
    const binding = TAURI_METHOD_REGISTRY[method];
    if (!binding) {
      throw new TauriRpcError(
        'METHOD_NOT_FOUND',
        `Method '${method}' is not registered in the Tauri bridge`,
        { method },
      );
    }
    // For chat.send, attach the per-session `agent:stream:{sessionId}` listener
    // BEFORE invoking send_message so the turn_start event is not missed. The
    // chat composables register their `session.event.*` handlers via `on()`
    // before the send, so the listener has a dispatch target.
    if (method === 'chat.send') {
      const sessionId = firstParam(params, 'sessionKey', 'session_key', 'sessionId', 'session_id');
      if (typeof sessionId === 'string' && sessionId) {
        this._ensureSessionStream(sessionId);
      }
    }
    const args = binding.transform ? binding.transform(params) : { ...params };
    const promise = Promise.resolve()
      .then(() => (binding.run ? binding.run(params) : invoke(binding.command, args)))
      .catch((err) => {
        throw normalizeTauriError(method, err);
      });
    return applyCallOptions(promise, method, options);
  }

  on(event: string, handler: RpcEventHandler): () => void {
    if (!this._listeners.has(event)) this._listeners.set(event, new Set());
    this._listeners.get(event)!.add(handler);

    if (!isInternalEvent(event)) {
      if (event === '*') {
        for (const name of TAURI_TOP_LEVEL_EVENTS) this._ensureDirectListener(name);
      } else if (isStreamingDiscriminator(event)) {
        // Streaming payloads arrive on the per-session `agent:stream:{sessionId}`
        // channel, which `chat.send` attaches before invoking `send_message`.
        // No global listener is needed here.
      } else {
        this._ensureDirectListener(event);
      }
    }

    return () => {
      const set = this._listeners.get(event);
      if (!set) return;
      set.delete(handler);
      if (set.size === 0) {
        this._listeners.delete(event);
        if (isInternalEvent(event)) {
          // Synthetic events have no Tauri listener to release.
        } else if (event === '*') {
          for (const name of TAURI_TOP_LEVEL_EVENTS) this._releaseDirectListener(name);
        } else if (!isStreamingDiscriminator(event)) {
          this._releaseDirectListener(event);
        }
      }
    };
  }

  get state(): ConnectionState {
    return this._state;
  }

  get policy(): Record<string, unknown> {
    return this._policy;
  }

  waitForConnection(
    timeoutMs: number = 30000,
    signal?: AbortSignal,
    _actions: RpcConnectionWaitOptions = {},
  ): Promise<void> {
    if (signal?.aborted) {
      return Promise.reject(new RpcAbortError('waitForConnection'));
    }
    if (this._state === 'connected') return Promise.resolve();

    return new Promise((resolve, reject) => {
      let settled = false;
      let timer: ReturnType<typeof setTimeout> | null = null;
      let off: () => void = () => {};

      const cleanup = (): void => {
        if (timer !== null) {
          clearTimeout(timer);
          timer = null;
        }
        off();
        signal?.removeEventListener('abort', onAbort);
      };
      const finish = (error?: Error): void => {
        if (settled) return;
        settled = true;
        cleanup();
        if (error) reject(error);
        else resolve();
      };
      const onAbort = (): void => {
        // Under Tauri there is no socket generation to recycle, so the
        // `timeoutAction` / `abortAction` options are honored only as `reject`.
        finish(new RpcAbortError('waitForConnection'));
      };

      off = this.on('_state', (s: ConnectionState) => {
        if (s === 'connected') finish();
      });
      signal?.addEventListener('abort', onAbort, { once: true });
      if (timeoutMs > 0 && Number.isFinite(timeoutMs)) {
        timer = setTimeout(() => {
          finish(new RpcTimeoutError('waitForConnection', timeoutMs));
        }, timeoutMs);
      }
    });
  }

  /**
   * Attach (once per session) a listener on `agent:stream:{sessionId}` and
   * route each `MessageStreamEvent` to the registered `session.event.*`
   * handlers via {@link mapMessageStreamEvent}.
   */
  private _ensureSessionStream(sessionId: string): void {
    if (this._sessionStreams.has(sessionId)) return;
    const unlisten = listen<MessageStreamEvent>(agentStreamChannel(sessionId), (payload) => {
      for (const { name, data } of mapMessageStreamEvent(payload)) {
        this._dispatch(name, data);
        this._dispatch('*', name, data, {});
      }
    });
    this._sessionStreams.set(sessionId, unlisten);
  }

  /**
   * Acquire a top-level Tauri channel listener. The RPC event name is mapped to
   * the actual Tauri channel via {@link TAURI_EVENT_CHANNEL} (e.g.
   * `sessions.changed` → `sessions:list-changed`). The listener is created on
   * the first acquisition and shared; callers must pair every acquisition with
   * {@link _releaseDirectListener}.
   */
  private _ensureDirectListener(event: string): void {
    const refs = (this._directRefs.get(event) ?? 0) + 1;
    this._directRefs.set(event, refs);
    if (this._directUnlisteners.has(event)) return;
    const channel = TAURI_EVENT_CHANNEL[event] ?? event;
    const unlisten = listen(channel, (payload: unknown) => {
      this._dispatch(event, payload);
      this._dispatch('*', event, payload, {});
    });
    this._directUnlisteners.set(event, unlisten);
  }

  /** Release a direct listener held by a subscription; tears down at zero. */
  private _releaseDirectListener(event: string): void {
    const refs = (this._directRefs.get(event) ?? 1) - 1;
    if (refs <= 0) {
      this._directRefs.delete(event);
      const unlisten = this._directUnlisteners.get(event);
      if (unlisten) {
        unlisten();
        this._directUnlisteners.delete(event);
      }
      return;
    }
    this._directRefs.set(event, refs);
  }

  /** Dispatch to every handler registered for `event`, then return. */
  private _dispatch(event: string, ...args: unknown[]): void {
    const handlers = this._listeners.get(event);
    if (!handlers || handlers.size === 0) return;
    for (const handler of [...handlers]) {
      try {
        handler(...args);
      } catch (error) {
        console.error(`[RPC] handler error for '${event}':`, error);
      }
    }
  }

  private _setState(state: ConnectionState): void {
    if (this._state === state) return;
    this._state = state;
    this._dispatch('_state', state);
  }
}
