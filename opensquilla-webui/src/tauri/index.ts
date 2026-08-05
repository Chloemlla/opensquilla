/**
 * Tauri bridge barrel export + initialization.
 *
 * This module is the single entry point for chat composables and views that
 * want to talk to the Rust runtime. It re-exports every adapter and provides
 * {@link initTauriBridge}, which:
 *
 *  - Detects whether the app is running inside a Tauri shell
 *    (`window.__TAURI__` exists).
 *  - When in Tauri: returns a bridge backed entirely by `invoke()` and
 *    Tauri events.
 *  - When NOT in Tauri (browser dev server, vitest): returns a bridge backed
 *    by the existing WebSocket/HTTP transport (`src/lib/rpc.ts` +
 *    `src/stores/rpc.ts`), so the same adapter API works in dev without a
 *    Tauri shell.
 *
 * This dual-mode design is the P1 "前端平台适配层" from
 * `docs/tauri-migration-analysis.md`: a thin adapter that lets the Vue 3 WebUI
 * keep its 632 source files unchanged while the transport is swapped from
 * WebSocket RPC to Tauri invoke/events.
 */

import { isTauri } from './invoke'
import type {
  TauriCoreApi,
  TauriGlobal,
} from './invoke'

export * from './invoke'
export * from './events'
export * from './gateway'
export * from './config'
export * from './providers'
export * from './system'

/* ──────────────────────────────────────────────────────────────────────────
 * Transport mode. Adapters and consumers branch on this to decide whether to
 * hit invoke() or the legacy WebSocket store.
 * ────────────────────────────────────────────────────────────────────────── */

export type TransportMode = 'tauri' | 'websocket'

/** The transport the bridge resolved to after the last init call. */
let resolvedMode: TransportMode | null = null

/**
 * The transport mode the bridge is using. Callers should prefer
 * {@link initTauriBridge} (which sets this) over reading the raw `isTauri()`
 * flag, because the bridge may be forced into websocket mode for testing.
 */
export function getTransportMode(): TransportMode {
  if (resolvedMode) return resolvedMode
  return isTauri() ? 'tauri' : 'websocket'
}

/* ──────────────────────────────────────────────────────────────────────────
 * WebSocket fallback.
 *
 * When not in Tauri, the bridge delegates to the existing Pinia RPC store
 * (`src/stores/rpc.ts`), which owns the WebSocket lifecycle. We import it
 * lazily (dynamic import) so that the Tauri production bundle — which never
 * needs the WebSocket client — can tree-shake it out. The lazy import is
 * guarded by `isTauri()` so the static analysis never pulls `@/stores/rpc`
 * into the Tauri build.
 * ────────────────────────────────────────────────────────────────────────── */

export interface WebSocketFallback {
  /** The Pinia RPC store (typed loosely to avoid a hard cycle). */
  readonly rpc: unknown
  /** True once the WebSocket has connected at least once. */
  readonly ready: boolean
}

let wsFallback: WebSocketFallback | null = null

async function loadWebSocketFallback(): Promise<WebSocketFallback> {
  if (wsFallback) return wsFallback
  // Lazy import keeps the WebSocket client out of the Tauri production bundle.
  const mod = await import('@/stores/rpc')
  const rpc = mod.useRpcStore()
  wsFallback = { rpc, ready: rpc.isConnected }
  return wsFallback
}

/* ──────────────────────────────────────────────────────────────────────────
 * initTauriBridge
 * ────────────────────────────────────────────────────────────────────────── */

export interface TauriBridgeInitOptions {
  /**
   * Force a specific transport, overriding the automatic `__TAURI__` detection.
   * Useful in tests (force `websocket` against a mock) and during staged
   * rollout (force `tauri` even when the global is slow to inject).
   */
  forceTransport?: TransportMode
  /**
   * Called once the bridge has resolved its transport. Receives the mode so
   * callers can wire up dev-mode banners.
   */
  onReady?: (mode: TransportMode) => void
}

export interface TauriBridge {
  /** The transport this bridge is using. */
  readonly mode: TransportMode
  /** True when the bridge is backed by Tauri invoke/events. */
  readonly isTauri: boolean
  /** The raw Tauri core API, or null in websocket mode. */
  readonly core: TauriCoreApi | null
  /** The WebSocket fallback handle, or null in tauri mode. */
  readonly websocket: WebSocketFallback | null
}

/**
 * Initialize the Tauri bridge and resolve the transport.
 *
 * In Tauri mode this is a near-noop: it confirms `window.__TAURI__.core`
 * exists and records the mode. In websocket mode it lazily loads the Pinia
 * RPC store so the rest of the app can subscribe to its connection state.
 *
 * Call this once during app bootstrap (e.g. in `main.ts` before
 * `app.mount()`), then use the exported adapters — they will route through
 * whichever transport this resolved.
 *
 * @example
 *   const bridge = await initTauriBridge()
 *   if (bridge.mode === 'websocket') console.warn('Running without Tauri shell')
 */
export async function initTauriBridge(
  options: TauriBridgeInitOptions = {},
): Promise<TauriBridge> {
  const tauriPresent = isTauri()
  const mode: TransportMode =
    options.forceTransport ?? (tauriPresent ? 'tauri' : 'websocket')
  resolvedMode = mode

  let core: TauriCoreApi | null = null
  let websocket: WebSocketFallback | null = null

  if (mode === 'tauri') {
    const tauri = (globalThis as unknown as TauriGlobal).__TAURI__
    core = tauri?.core ?? null
    if (!core) {
      // The global claimed to be Tauri but the core API is missing — fall
      // back to websocket so the app stays usable instead of throwing on
      // every invoke.
      resolvedMode = 'websocket'
      websocket = await loadWebSocketFallback()
    }
  } else {
    websocket = await loadWebSocketFallback()
  }

  options.onReady?.(resolvedMode)

  return {
    mode: resolvedMode,
    isTauri: resolvedMode === 'tauri',
    core,
    websocket,
  }
}

/** Reset the bridge state. Intended for tests only. */
export function __resetTauriBridgeForTests(): void {
  resolvedMode = null
  wsFallback = null
}
