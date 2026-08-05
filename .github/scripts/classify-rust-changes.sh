#!/usr/bin/env bash
set -euo pipefail

# Classify Rust changes for selective CI
changed_files="${1:-}"

if [[ ! -f "${changed_files}" ]]; then
  echo "rust_changed=false"
  echo "rust_core_changed=false"
  echo "rust_gateway_changed=false"
  echo "rust_engine_changed=false"
  echo "rust_provider_changed=false"
  exit 0
fi

rust_changed=false
rust_core_changed=false
rust_gateway_changed=false
rust_engine_changed=false
rust_provider_changed=false

while IFS= read -r file; do
  case "${file}" in
    Cargo.toml|rust-toolchain.toml|clippy.toml)
      rust_changed=true
      rust_core_changed=true
      rust_gateway_changed=true
      rust_engine_changed=true
      rust_provider_changed=true
      ;;
    crates/core/*)
      rust_changed=true
      rust_core_changed=true
      ;;
    crates/gateway/*)
      rust_changed=true
      rust_gateway_changed=true
      ;;
    crates/engine/*)
      rust_changed=true
      rust_engine_changed=true
      ;;
    crates/provider/*)
      rust_changed=true
      rust_provider_changed=true
      ;;
    crates/tools/*|crates/session/*|crates/memory/*|crates/channels/*)
      rust_changed=true
      ;;
    crates/*)
      rust_changed=true
      ;;
  esac
done < "${changed_files}"

echo "rust_changed=${rust_changed}"
echo "rust_core_changed=${rust_core_changed}"
echo "rust_gateway_changed=${rust_gateway_changed}"
echo "rust_engine_changed=${rust_engine_changed}"
echo "rust_provider_changed=${rust_provider_changed}"