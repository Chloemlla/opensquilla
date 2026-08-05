#!/usr/bin/env node
/**
 * build-frontend.mjs — Build the Vue 3 WebUI for the Tauri shell.
 *
 * Invoked by `tauri.conf.json` as `beforeBuildCommand` / `beforeDevCommand`
 * so that `cargo tauri build` / `cargo tauri dev` always has a fresh
 * frontend bundle in the location `frontendDist` points at.
 *
 * Responsibilities:
 *   1. Run `npm ci` (or `npm install` as a fallback) in opensquilla-webui.
 *   2. Run `npm run build` (or `npm run dev` in --dev mode, which just primes
 *      the Vite dev server cache — the dev server itself is started by Tauri
 *      via `devUrl`).
 *   3. Verify the expected dist output exists.
 *   4. (No-op copy step placeholder for any additional platform assets that
 *      need to land beside the bundle — currently empty, the Vite config
 *      already emits everything into dist/).
 *
 * Usage:
 *   node scripts/build-frontend.mjs           # production build
 *   node scripts/build-frontend.mjs --build   # explicit production build
 *   node scripts/build-frontend.mjs --dev     # dev mode (primes deps only)
 *
 * Exits non-zero on any failure so `cargo tauri build` aborts instead of
 * packaging a stale or empty frontend.
 */

import { spawnSync } from 'node:child_process'
import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs'
import { resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)
const repoRoot = resolve(__dirname, '..')
const webuiDir = resolve(repoRoot, 'opensquilla-webui')

// --- args -------------------------------------------------------------------

const args = process.argv.slice(2)
const devMode = args.includes('--dev') && !args.includes('--build')

// --- helpers ----------------------------------------------------------------

/**
 * Run a command, inheriting stdio so npm's output streams to the console.
 * Throws (and aborts the process) on non-zero exit.
 */
function run(cmd, cmdArgs, cwd) {
  const display = `${cmd} ${cmdArgs.join(' ')}`
  console.log(`> ${display}`)
  const result = spawnSync(cmd, cmdArgs, {
    cwd,
    stdio: 'inherit',
    shell: process.platform === 'win32',
  })
  if (result.error) {
    throw new Error(`Failed to spawn "${display}": ${result.error.message}`)
  }
  if (result.status !== 0) {
    throw new Error(`"${display}" exited with code ${result.status}`)
  }
}

/**
 * Pick the npm binary name. On Windows the shim is `npm.cmd` when not going
 * through a shell; with `shell: true` above, plain `npm` resolves correctly.
 */
const npmBin = 'npm'

// --- preflight --------------------------------------------------------------

if (!existsSync(webuiDir)) {
  console.error(`ERROR: WebUI directory not found: ${webuiDir}`)
  process.exit(1)
}

const pkgJsonPath = resolve(webuiDir, 'package.json')
if (!existsSync(pkgJsonPath)) {
  console.error(`ERROR: package.json not found in ${webuiDir}`)
  process.exit(1)
}

const nodeVersionFile = resolve(webuiDir, '.node-version')
if (existsSync(nodeVersionFile)) {
  // Just surface the required version; setup-node in CI enforces it.
  const requiredNode = readFileSync(nodeVersionFile, 'utf8').trim()
  console.log(`WebUI requires Node ${requiredNode} (per .node-version)`)
  console.log(`Running on Node ${process.versions.node}`)
}

// --- install deps ------------------------------------------------------------

console.log(`\n[1/3] Installing WebUI dependencies in ${webuiDir}`)

const lockfile = resolve(webuiDir, 'package-lock.json')
const nodeModulesDir = resolve(webuiDir, 'node_modules')

if (devMode && existsSync(nodeModulesDir)) {
  // Dev mode: reuse the existing install so `cargo tauri dev` starts fast.
  // `npm ci` would wipe node_modules on every dev launch.
  console.log('node_modules already present; skipping install in dev mode')
} else if (existsSync(lockfile)) {
  // `npm ci` is the reproducible install for release builds — it refuses to
  // touch package-lock.json and fails fast if it is out of sync.
  run(npmBin, ['ci'], webuiDir)
} else {
  console.warn('package-lock.json missing; falling back to `npm install` (non-reproducible)')
  run(npmBin, ['install'], webuiDir)
}

// --- build ------------------------------------------------------------------

if (devMode) {
  // In dev mode Tauri starts the Vite dev server itself (via devUrl). We only
  // need deps installed (above) so `npm run dev` can launch without fetching.
  console.log('\n[2/3] Dev mode: skipping production build (Tauri will start the Vite dev server)')
  console.log('\n[3/3] Dev mode: nothing to verify')
  process.exit(0)
}

console.log('\n[2/3] Building WebUI for production')
run(npmBin, ['run', 'build'], webuiDir)

// --- verify dist ------------------------------------------------------------

// The Vite config (opensquilla-webui/vite.config.ts) emits to this outDir.
// tauri.conf.json -> build.frontendDist points at the same path (resolved
// relative to src-tauri/), so they must stay in sync.
const distDir = resolve(webuiDir, 'dist')
const indexHtml = resolve(distDir, 'index.html')

console.log('\n[3/3] Verifying build output')

if (!existsSync(distDir) || !statSync(distDir).isDirectory()) {
  console.error(`ERROR: dist directory was not produced: ${distDir}`)
  console.error('       Check the vite build output above for errors.')
  process.exit(1)
}

if (!existsSync(indexHtml)) {
  console.error(`ERROR: ${indexHtml} is missing — the Vite build produced no entrypoint.`)
  console.error('       The Tauri shell cannot serve the WebUI without dist/index.html.')
  process.exit(1)
}

// Surface the artifact tree so release logs show what got bundled.
const topLevel = readdirSync(distDir).sort()
console.log(`\nWebUI dist/ contains ${topLevel.length} top-level entries:`)
for (const entry of topLevel) {
  console.log(`  - ${entry}`)
}

console.log('\nWebUI build OK. Tauri can now bundle the frontend.')
