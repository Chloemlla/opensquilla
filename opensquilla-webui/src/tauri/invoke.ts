/**
 * Tauri invoke() core wrapper.
 *
 * This is the single chokepoint through which every Tauri adapter in
 * `opensquilla-webui/src/tauri/` reaches the Rust runtime. It wraps
 * `window.__TAURI__.core.invoke()` with:
 *
 *  - A Tauri-context guard (throws a typed error outside Tauri, so callers can
 *    fall back to the dev-mode HTTP/WebSocket transport via {@link isTauri}).
 *  - Normalized error handling: Tauri rejects with either a string or an object
 *    shaped like `{ code?, message?, details? }`. We surface those as
 *    {@link TauriInvokeError} so adapters can switch on `code` without
 *    re-parsing the rejection.
 *  - Strongly typed generics so each adapter declares the exact return shape.
 *
 * The Rust `tauri::command` signatures these calls target are defined in the
 * Tauri shell crate (see `docs/tauri-migration-analysis.md` P1). The argument
 * objects here mirror the serde struct field names (camelCase keys are
 * forwarded as-is; serde is configured with `#[serde(rename_all = "camelCase")]`
 * on the Rust side for IPC payloads).
 */

/** Shape of the Tauri global injected by the shell. Matches @tauri-apps/api v2. */
export interface TauriCoreApi {
  invoke<T = unknown>(cmd: string, args?: Record<string, unknown>): Promise<T>
}

export interface TauriGlobal {
  __TAURI__?: {
    core?: TauriCoreApi
  }
}

/** Error detail mirroring the Rust `IpcError` serde struct returned on reject. */
export interface TauriErrorDetail {
  /** Stable machine code, e.g. `SESSION_NOT_FOUND`, `CONFIG_INVALID`. */
  code?: string
  /** Human-readable message. */
  message?: string
  /** Optional structured details (serde_json::Value on the Rust side). */
  details?: unknown
  /** Whether the operation can be retried (network blip, lock contention). */
  retryable?: boolean
  /** Suggested backoff in milliseconds before retrying. */
  retry_after_ms?: number
}

/**
 * Normalized error thrown by {@link invoke} / {@link invokeCommand} when a
 * Tauri command rejects. Adapters should `throw` this onward unchanged so UI
 * layers can read `.code` for branching.
 */
export class TauriInvokeError extends Error {
  readonly code: string | undefined
  readonly details: unknown
  readonly retryable: boolean
  readonly retry_after_ms: number | undefined
  /** The Tauri command that failed, for diagnostics. */
  readonly command: string
  /** The original rejection (Error or raw value), for inspection. */
  readonly cause: unknown

  constructor(command: string, cause: unknown) {
    const detail = normalizeErrorPayload(cause)
    super(detail.message || `Tauri command '${command}' failed`)
    this.name = 'TauriInvokeError'
    this.command = command
    this.code = detail.code
    this.details = detail.details
    this.retryable = detail.retryable ?? false
    this.retry_after_ms = detail.retry_after_ms

    // Preserve the original rejection (Error or raw value) for inspection
    // without altering the stack.
    this.cause = cause
  }
}

function normalizeErrorPayload(cause: unknown): TauriErrorDetail {
  if (typeof cause === 'string' && cause.length > 0) {
    return { message: cause }
  }
  if (cause && typeof cause === 'object') {
    const raw = cause as Record<string, unknown>
    const message =
      typeof raw.message === 'string' && raw.message.length > 0
        ? raw.message
        : typeof raw.error === 'string' && raw.error.length > 0
          ? raw.error
          : undefined
    return {
      code: typeof raw.code === 'string' && raw.code.length > 0 ? raw.code : undefined,
      message,
      details: raw.details,
      retryable: typeof raw.retryable === 'boolean' ? raw.retryable : undefined,
      retry_after_ms:
        typeof raw.retry_after_ms === 'number' && Number.isFinite(raw.retry_after_ms)
          ? raw.retry_after_ms
          : typeof raw.retryAfterMs === 'number' && Number.isFinite(raw.retryAfterMs)
            ? raw.retryAfterMs
            : undefined,
    }
  }
  return { message: cause === undefined ? 'Unknown Tauri error' : String(cause) }
}

/**
 * Read the Tauri core API from the global. Returns `null` when not running
 * inside a Tauri shell (browser dev server, vitest with happy-dom, etc.).
 */
function getTauriCore(): TauriCoreApi | null {
  const tauri = (globalThis as unknown as TauriGlobal).__TAURI__
  return tauri?.core ?? null
}

/** True when running inside a Tauri shell (i.e. `window.__TAURI__` exists). */
export function isTauri(): boolean {
  return getTauriCore() !== null
}

/**
 * Require the Tauri core API, throwing a typed error if it is missing. Adapters
 * call this to fail fast with an actionable message instead of a bare
 * `Cannot read properties of undefined`.
 */
function requireTauriCore(command: string): TauriCoreApi {
  const core = getTauriCore()
  if (!core) {
    throw new TauriInvokeError(
      command,
      {
        code: 'TAURI_UNAVAILABLE',
        message:
          `Tauri runtime is not available for command '${command}'. ` +
          `Running outside a Tauri shell — use initTauriBridge() to fall back to HTTP/WebSocket.`,
        retryable: false,
      },
    )
  }
  return core
}

/**
 * Typed Tauri invoke wrapper.
 *
 * @typeParam T - The deserialized return type of the command (mirrors the Rust
 *                `tauri::command` return `Result<T, IpcError>`).
 * @param command - The Rust command name, e.g. `'get_config'`, `'send_message'`.
 * @param args - Argument object; keys mirror the Rust serde struct field names.
 * @returns The resolved payload, typed as `T`.
 * @throws {TauriInvokeError} on any rejection or when not in a Tauri context.
 */
export async function invoke<T = unknown>(
  command: string,
  args?: Record<string, unknown>,
): Promise<T> {
  const core = requireTauriCore(command)
  try {
    return await core.invoke<T>(command, args)
  } catch (cause) {
    throw new TauriInvokeError(command, cause)
  }
}

/**
 * Generic command helper. Identical to {@link invoke} but explicit about the
 * generic surface; adapters use it to bind a command name to a fixed return
 * type at the call site.
 *
 * @example
 *   const cfg = await invokeCommand<GatewayConfig>('get_config')
 */
export function invokeCommand<T = unknown>(
  cmd: string,
  args?: Record<string, unknown>,
): Promise<T> {
  return invoke<T>(cmd, args)
}

/**
 * Invoke a command that may legitimately be missing on older shells. Returns
 * `null` instead of throwing when the shell reports `COMMAND_NOT_FOUND` (the
 * Rust side returns this code for unregistered commands during the migration
 * window). Useful for capability-gated adapters like native update checks.
 */
export async function invokeOptional<T = unknown>(
  command: string,
  args?: Record<string, unknown>,
): Promise<T | null> {
  try {
    return await invoke<T>(command, args)
  } catch (err) {
    if (err instanceof TauriInvokeError && err.code === 'COMMAND_NOT_FOUND') {
      return null
    }
    throw err
  }
}

