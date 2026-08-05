/**
 * tauri-plugin-shim.ts
 *
 * Development shim for the `window.__TAURI__` global.
 *
 * The WebUI (Vue 3, built by Vite + vue-tsc) is compiled standalone — `npm
 * run build` must succeed even when there is no Tauri shell injecting
 * `window.__TAURI__`. This file:
 *
 *   1. Declares the ambient `__TAURI__` global (ambient module augmentation)
 *      so TypeScript knows the shape of the API `@tauri-apps/api` v2 exposes,
 *      without adding `@tauri-apps/api` as a dependency.
 *   2. Exports an *opt-in* `installTauriMock()` that installs a no-op mock of
 *      `window.__TAURI__` for local browser development and unit tests.
 *
 * IMPORTANT: the mock is intentionally NOT installed automatically. The app
 * detects the Tauri context by checking whether `window.__TAURI__.core`
 * exists (`src/tauri/invoke.ts` -> `isTauri()`). If a fake global were
 * installed eagerly, the bridge would take the `tauri` transport and every
 * `invoke()` would fail. Only call `installTauriMock()` in code that is
 * explicitly running a mock/dev harness.
 *
 * The main tsconfig.json includes this file (see `include`), which makes the
 * ambient global visible to the whole `src/` tree during `vue-tsc --noEmit`.
 */

/* ──────────────────────────────────────────────────────────────────────────
 * Ambient global declaration.
 *
 * Mirrors the `@tauri-apps/api` v2 surface. Only the members the WebUI
 * actually uses are enumerated — extend as the app grows.
 * ────────────────────────────────────────────────────────────────────────── */

declare global {
  interface TauriEvent<T = unknown> {
    /** Event name, e.g. "gateway://status". */
    event: string
    /** Id of the event emitter. */
    id: number
    /** Payload emitted from the Rust side. */
    payload: T
  }

  interface TauriCoreApi {
    /**
     * Invoke a command registered by the Rust shell
     * (`tauri::generate_handler![...]`).
     */
    invoke<T = unknown>(
      cmd: string,
      args?: Record<string, unknown>,
    ): Promise<T>
    /** Convert an IPC payload into a byte array (used rarely). */
    transformCallback?<T>(callback: (response: T) => void): number
  }

  interface TauriEventApi {
    /** Subscribe to a Rust-emitted event. Returns an unsubscribe function. */
    listen<T = unknown>(
      event: string,
      handler: (event: TauriEvent<T>) => void,
    ): Promise<() => void>
    /** One-shot subscription to the next occurrence of an event. */
    once<T = unknown>(
      event: string,
      handler: (event: TauriEvent<T>) => void,
    ): Promise<() => void>
    /** Emit an event to the Rust side (or to other webviews). */
    emit(event: string, payload?: unknown): Promise<void>
  }

  interface TauriWindowApi {
    label: string
    minimize(): Promise<void>
    maximize(): Promise<void>
    unmaximize(): Promise<void>
    toggleMaximize(): Promise<void>
    show(): Promise<void>
    hide(): Promise<void>
    close(): Promise<void>
    setTitle(title: string): Promise<void>
    isMinimized(): Promise<boolean>
    isMaximized(): Promise<boolean>
    isVisible(): Promise<boolean>
    isFocused(): Promise<boolean>
    onResized(handler: (event: TauriEvent<{ width: number; height: number }>) => void): Promise<() => void>
    onMoved(handler: (event: TauriEvent<{ x: number; y: number }>) => void): Promise<() => void>
    onCloseRequested(handler: (event: TauriEvent<unknown>) => void): Promise<() => void>
  }

  interface TauriAppApi {
    getVersion(): Promise<string>
    getName(): Promise<string>
    getTauriVersion(): Promise<string>
    exit(code?: number): Promise<void>
    relaunch(): Promise<void>
  }

  interface TauriOsApi {
    platform(): Promise<string>
    version(): Promise<string>
    family(): Promise<string>
    arch(): Promise<string>
    type(): Promise<string>
  }

  interface TauriPathApi {
    appConfigDir(): Promise<string>
    appDataDir(): Promise<string>
    appLogDir(): Promise<string>
    desktopDir(): Promise<string>
    documentDir(): Promise<string>
    downloadDir(): Promise<string>
    homeDir(): Promise<string>
    resourceDir(): Promise<string>
    tempDir(): Promise<string>
  }

  interface TauriPluginShim {
    core?: TauriCoreApi
    event?: TauriEventApi
    window?: TauriWindowApi
    app?: TauriAppApi
    os?: TauriOsApi
    path?: TauriPathApi
    /** Raw command invocation used by the plugin bridge. */
    invoke?: TauriCoreApi['invoke']
  }

  interface Window {
    __TAURI__?: TauriPluginShim
  }
}

export {}

/* ──────────────────────────────────────────────────────────────────────────
 * Re-exported types for consumers that want to annotate against the shim
 * without importing `@tauri-apps/api`.
 * ────────────────────────────────────────────────────────────────────────── */

export type TauriGlobal = Window['__TAURI__']
export type TauriCore = NonNullable<TauriGlobal>['core']
export type TauriEventLike<T = unknown> = TauriEvent<T>

/** True when a Tauri shell has injected `window.__TAURI__.core`. */
export function hasTauriContext(): boolean {
  return typeof window !== 'undefined' && Boolean(window.__TAURI__?.core)
}

/* ──────────────────────────────────────────────────────────────────────────
 * Opt-in dev mock.
 *
 * Call `installTauriMock()` from a dev-only entry (e.g. a storybook/preview
 * harness or a `mock.ts` you import in vitest setup) to stand up a working
 * fake. Every method resolves; `invoke` rejects unless a handler is
 * registered with `registerMockHandler`.
 * ────────────────────────────────────────────────────────────────────────── */

export interface TauriMockOptions {
  /** Version strings returned by `app.getVersion()` etc. */
  appVersion?: string
  appName?: string
  /** Platform reported by `os.platform()`. */
  platform?: string
  /** Replace specific API members wholesale. */
  overrides?: Partial<TauriPluginShim>
}

const handlers = new Map<string, (args: Record<string, unknown>) => unknown>()

/**
 * Register a fake command handler for `installTauriMock()`'s `invoke`.
 * In tests, register commands your component calls before invoking them.
 */
export function registerMockHandler(
  command: string,
  handler: (args: Record<string, unknown>) => unknown,
): void {
  handlers.set(command, handler)
}

/** Remove all registered mock command handlers. */
export function resetMockHandlers(): void {
  handlers.clear()
}

/**
 * Install a no-op `window.__TAURI__` for dev/test environments. Calling this
 * makes `hasTauriContext()`/`isTauri()` return true, so only call it in
 * harnesses that provide mock command handlers for every invoke the app makes.
 */
export function installTauriMock(options: TauriMockOptions = {}): TauriPluginShim {
  const api: TauriPluginShim = {
    core: {
      invoke: async <T>(cmd: string, args: Record<string, unknown> = {}): Promise<T> => {
        const handler = handlers.get(cmd)
        if (!handler) {
          throw {
            code: 'COMMAND_NOT_FOUND',
            message: `No mock handler registered for command '${cmd}'. ` +
              `Call registerMockHandler('${cmd}', ...) before invoking.`,
            retryable: false,
          }
        }
        return handler(args) as T
      },
    },
    event: {
      listen: async <T>(event: string, _handler: (event: TauriEvent<T>) => void) => {
        // No events are emitted in the mock; return an inert unsubscribe.
        return () => undefined
      },
      once: async <T>(_event: string, _handler: (event: TauriEvent<T>) => void) => {
        return () => undefined
      },
      emit: async (_event: string, _payload?: unknown) => undefined,
    },
    window: {
      label: 'main',
      minimize: async () => undefined,
      maximize: async () => undefined,
      unmaximize: async () => undefined,
      toggleMaximize: async () => undefined,
      show: async () => undefined,
      hide: async () => undefined,
      close: async () => undefined,
      setTitle: async () => undefined,
      isMinimized: async () => false,
      isMaximized: async () => false,
      isVisible: async () => true,
      isFocused: async () => true,
      onResized: async () => () => undefined,
      onMoved: async () => () => undefined,
      onCloseRequested: async () => () => undefined,
    },
    app: {
      getVersion: async () => options.appVersion ?? '0.0.0-dev',
      getName: async () => options.appName ?? 'OpenSquilla',
      getTauriVersion: async () => '2.0.0',
      exit: async () => undefined,
      relaunch: async () => undefined,
    },
    os: {
      platform: async () => options.platform ?? 'development',
      version: async () => '0.0.0',
      family: async () => 'unknown',
      arch: async () => 'unknown',
      type: async () => 'development',
    },
    path: {
      appConfigDir: async () => '.',
      appDataDir: async () => '.',
      appLogDir: async () => '.',
      desktopDir: async () => '.',
      documentDir: async () => '.',
      downloadDir: async () => '.',
      homeDir: async () => '.',
      resourceDir: async () => '.',
      tempDir: async () => '.',
    },
    invoke: async <T>(cmd: string, args?: Record<string, unknown>) =>
      (api.core!.invoke as (c: string, a: Record<string, unknown>) => Promise<T>)(cmd, args ?? {}),
  }

  window.__TAURI__ = { ...api, ...options.overrides }
  return window.__TAURI__
}
