#!/usr/bin/env bash
# Generate a Tauri v2 updater keypair for signing and verifying update bundles.
#
# The Tauri updater plugin uses a minisign-style keypair: the private key signs
# each release's update artifact (the `.tar.gz` produced when
# `bundle.createUpdaterArtifacts` is true), and the public key is embedded in
# `tauri.conf.json` under `plugins.updater.pubkey` so the installed app can
# verify updates before applying them.
#
# This script generates a fresh keypair and prints instructions for wiring it
# into the release pipeline. Run it once per project (or once per signing key
# rotation). NEVER commit the private key.
#
# Usage:
#   scripts/generate-updater-key.sh [private-key-output-path]
#
# Defaults:
#   private key  -> ./opensquilla-updater.key   (current dir; do NOT commit)
#   public key   -> printed to stdout
#
# PowerShell equivalent (Windows, no sh):
#   $env:TAURI_PRIVATE_KEY_PASSWORD = "your-strong-password"
#   cargo tauri signer generate -w "$PWD\opensquilla-updater.key"
#   Get-Content "$PWD\opensquilla-updater.key.pub"
#
# Requires the Tauri CLI:
#   cargo install tauri-cli --version "^2"
set -euo pipefail

# --- preflight --------------------------------------------------------------

if ! command -v cargo >/dev/null 2>&1; then
  echo "ERROR: cargo is not on PATH. Install the Rust toolchain first." >&2
  exit 1
fi

# Ensure the Tauri CLI is available; `cargo tauri` works whether the binary is
# installed globally or invoked through `cargo`.
if ! cargo tauri --version >/dev/null 2>&1; then
  echo "Tauri CLI not found. Installing tauri-cli v2..." >&2
  cargo install tauri-cli --version "^2" --locked
fi

# --- key path ---------------------------------------------------------------

KEY_PATH="${1:-./opensquilla-updater.key}"
KEY_DIR="$(dirname "$KEY_PATH")"
if [[ ! -d "$KEY_DIR" ]]; then
  mkdir -p "$KEY_DIR"
fi

# --- password ---------------------------------------------------------------

# The private key is encrypted with a password. Read it from the environment if
# the caller set TAURI_PRIVATE_KEY_PASSWORD; otherwise prompt (so the password
# never lands in the shell history).
if [[ -z "${TAURI_PRIVATE_KEY_PASSWORD:-}" ]]; then
  echo "Enter a strong password to encrypt the updater private key:" >&2
  read -r -s TAURI_PRIVATE_KEY_PASSWORD
  if [[ -z "$TAURI_PRIVATE_KEY_PASSWORD" ]]; then
    echo "ERROR: password is required (the private key must be encrypted)." >&2
    exit 1
  fi
  export TAURI_PRIVATE_KEY_PASSWORD
  echo "" >&2
fi

# --- generate ---------------------------------------------------------------

echo "Generating Tauri updater keypair..." >&2
echo "  private key -> $KEY_PATH" >&2
echo "  public key  -> $KEY_PATH.pub" >&2
echo "" >&2

# -w/--write-keys writes both keys to disk:
#   $KEY_PATH       (private, encrypted — KEEP SECRET)
#   $KEY_PATH.pub   (public — safe to publish)
cargo tauri signer generate -w "$KEY_PATH" --force

if [[ ! -f "$KEY_PATH" || ! -f "$KEY_PATH.pub" ]]; then
  echo "ERROR: key generation did not produce both key files." >&2
  exit 1
fi

# --- output -----------------------------------------------------------------

PUB_KEY="$(cat "$KEY_PATH.pub")"

echo ""
echo "================================================================"
echo " Tauri updater keypair generated"
echo "================================================================"
echo ""
echo "PUBLIC KEY (paste into src-tauri/tauri.conf.json ->"
echo "plugins.updater.pubkey, replacing the PLACEHOLDER_ value):"
echo ""
echo "  $PUB_KEY"
echo ""
echo "PRIVATE KEY (kept at $KEY_PATH):"
echo "  - Add it as a GitHub Actions repository secret named"
echo "    TAURI_PRIVATE_KEY (Settings -> Secrets and variables -> Actions)."
echo "  - Add the password as a secret named TAURI_PRIVATE_KEY_PASSWORD."
echo "  - The release workflow (.github/workflows/release.yml) reads these"
echo "    to sign update artifacts."
echo ""
echo "SECURITY:"
echo "  - The private key file ($KEY_PATH) is generated in the current"
echo "    directory. Add it to .gitignore or delete it after uploading the"
echo "    secret. NEVER commit it."
echo "  - Rotate the keypair by re-running this script with a new path and"
echo "    updating the pubkey in tauri.conf.json."
echo "================================================================"
