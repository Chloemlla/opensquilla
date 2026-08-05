# OpenSquilla — Token-Efficient AI Agent

<p align="center">
  <img src="assets/opensquilla-long-logo.png" alt="OpenSquilla logo" width="500">
</p>

<p align="center">
  <b>Same budget, more capability, better results.</b><br>
  A native Rust AI agent for desktop, CLI, and chat channels.
</p>

<p align="center">
  <a href="https://github.com/opensquilla/opensquilla/actions/workflows/rust-ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/opensquilla/opensquilla/rust-ci.yml?style=for-the-badge" alt="CI"></a>
  <a href="https://opensquilla.ai/"><img src="https://img.shields.io/badge/website-opensquilla.ai-blue?style=for-the-badge" alt="Website"></a>
  <a href="https://github.com/opensquilla/opensquilla/releases"><img src="https://img.shields.io/github/v/release/opensquilla/opensquilla?include_prereleases&style=for-the-badge" alt="GitHub release"></a>
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/rust-1.85%2B-orange?style=for-the-badge" alt="Rust"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue?style=for-the-badge" alt="Apache 2.0 License"></a>
</p>

<p align="center">
  <b>English</b> · <a href="README.zh-Hans.md">中文</a> · <a href="README.ja.md">日本語</a> · <a href="README.fr.md">Français</a> · <a href="README.de.md">Deutsch</a> · <a href="README.es.md">Español</a>
</p>

---

## Overview

OpenSquilla is a token-efficient AI agent built as a **native Rust desktop application** (Tauri v2). A local model router sends each turn to the cheapest model that can handle it, while persistent memory, a layered sandbox, built-in web search, and on-device embeddings round out a single shared turn loop.

Every entry point — desktop Web UI, CLI, TUI, and chat channels — runs through that same loop, so tool dispatch, retries, and decision logging behave identically everywhere. A pluggable provider layer speaks to TokenRhythm, OpenRouter, OpenAI, Anthropic, Ollama, DeepSeek, Gemini, Qwen/DashScope, and 50+ other LLM providers with no change to your code or config schema.

**Key differences from the Python version:**
- Single native binary — no Python runtime, no pip, no uv
- Tauri v2 desktop shell with system tray, deep links, and auto-updates
- All 29 Rust workspace crates in one process, zero sidecars
- Vue 3 WebUI embedded in the binary

---

## Quick Start

### Desktop (recommended)

Download the latest installer from the [Releases page](https://github.com/opensquilla/opensquilla/releases):

| Platform | Format |
|----------|--------|
| Windows | `.msi` / `.exe` (NSIS) |
| macOS | `.dmg` (Apple Silicon + Intel) |
| Linux | `.deb` / `.AppImage` |

Launch the app, run the onboarding wizard, and start chatting.

### CLI

```sh
osq onboard
osq chat
```

### TUI

```sh
osq-tui
```

---

## Build from Source

### Prerequisites

| Requirement | Version |
|-------------|---------|
| Rust | 1.85+ (see `rust-toolchain.toml`) |
| Node.js | 22.12+ (for WebUI build) |
| npm | 10+ |

**Linux** also requires system libraries for Tauri's WebView:

```sh
sudo apt-get install -y libwebkit2gtk-4.1-dev libxdo-dev libssl-dev \
  libayatana-appindicator3-dev librsvg2-dev patchelf
```

### Build

```sh
# Build the WebUI
cd opensquilla-webui
npm ci
npm run build
cd ..

# Build the desktop app
cargo tauri build --project-path src-tauri

# Or build CLI-only
cargo build -p opensquilla_cli
cargo build -p osq_tui
```

Build artifacts:
- `src-tauri/target/release/opensquilla.exe` — desktop app
- `target/release/osq.exe` — CLI
- `target/release/osq-tui.exe` — TUI

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  Tauri v2 Desktop Shell (Rust)                                   │
│  ├─ Window / Tray / Deep links / Auto-updater                    │
│  └─ Tauri commands → Rust Agent Runtime (same process)           │
├──────────────────────────────────────────────────────────────────┤
│  Rust Agent Runtime (29 workspace crates, single binary)         │
│  ├─ Gateway (axum HTTP/WS)                                       │
│  ├─ Agent Engine (tokio async state machine)                     │
│  ├─ 30+ RPC handlers                                             │
│  ├─ 50+ LLM provider adapters (reqwest + SSE)                    │
│  ├─ 22 built-in tools (tokio::process::Command)                  │
│  ├─ Session management (rusqlite) + Memory (FTS5 + candle)       │
│  ├─ 11 channel adapters (reqwest + tokio-tungstenite)            │
│  ├─ Security sandbox (nix + windows-rs)                          │
│  ├─ CLI / TUI (clap + ratatui)                                   │
│  └─ Skills system + Scheduler (tokio cron)                       │
└──────────────────────────────────────────────────────────────────┘
```

### Runtime

| Component | Rust Crate | Python Equivalent |
|-----------|-----------|-------------------|
| Async runtime | `tokio` | `asyncio` |
| HTTP server | `axum` | `Starlette` |
| HTTP client | `reqwest` | `httpx` |
| WebSocket | `tokio-tungstenite` | `websockets` |
| Serialization | `serde` | `Pydantic` |
| Database | `rusqlite` | `aiosqlite` |
| CLI | `clap` | `typer` |
| TUI | `ratatui` | `Rich` |
| Logging | `tracing` | `structlog` |

---

## Key Features

| Capability | Description |
| --- | --- |
| **Token-efficient routing** | Local SquillaRouter (ONNX + LightGBM) classifies each turn and routes to the cheapest capable model across four tiers (C0–C3). Classification runs on-device. |
| **50+ LLM providers** | TokenRhythm, OpenRouter, OpenAI, Anthropic, Ollama, DeepSeek, Gemini, DashScope/Qwen, Moonshot, Mistral, Groq, Zhipu, SiliconFlow, vLLM, LM Studio, and more. |
| **On-demand skills** | 15+ bundled skills (coding, GitHub, cron, document generation, summarization, weather, and more). MCP client + server support. |
| **Persistent memory** | SQLite FTS5 full-text search + `sqlite-vec` semantic recall. On-device ONNX embeddings, optional dream consolidation. |
| **Layered sandbox** | Bubblewrap (Linux), Seatbelt (macOS), Windows native (CreateRestrictedToken). Three policy tiers, denial ledger, output purging. |
| **22 built-in tools** | File read/write/edit, shell, git, web search (7 providers), web fetch, image/PDF/TTS, document authoring, memory tools, cron. |
| **11 channels** | Slack, Discord, Telegram, Feishu, DingTalk, QQ, WeCom, Matrix, MS Teams, Terminal, WebSocket. |
| **Unified gateway** | axum server on `127.0.0.1:18791` with WebSocket RPC. Desktop WebUI, CLI, TUI, and channels share one TurnRunner. |
| **Operator controls** | Human-in-the-loop approvals, per-turn/per-session token and cost rollups, diagnostics. |

---

## Configuration

### First-run setup

```sh
osq onboard                    # interactive wizard
osq onboard --if-needed        # idempotent
osq onboard --minimal          # provider only
```

### Run

```sh
osq chat                       # interactive REPL
osq agent -m "your prompt"     # one-shot
osq-tui                        # TUI mode
```

### Config

Config load order: `OPENSQUILLA_GATEWAY_CONFIG_PATH` → `./opensquilla.toml` → `~/.opensquilla/config.toml` → built-in defaults. See `opensquilla.toml.example` for all options.

---

## Benchmark Results

PinchBench 1.2.1 average results across 25 tasks:

| Agent | Base Model | Avg. score | Total input tokens | Total output tokens | Total cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| OpenSquilla | Model router (Opus4.7, GLM5.1, DS4 Flash) | 0.9251 | 1,721,328 | 61,475 | $0.688 |
| OpenClaw | Claude Opus 4.7 | 0.9255 | 3,066,243 | 50,890 | $6.233 |

---

## Project Structure

```
├── crates/             # 29 Rust workspace crates
│   ├── engine/         # Agent runtime, turn runner, routing
│   ├── gateway/        # axum HTTP/WS server, 30+ RPC handlers
│   ├── provider/       # 50+ LLM provider adapters
│   ├── session/        # Session management (rusqlite)
│   ├── memory/         # Memory system (FTS5 + sqlite-vec)
│   ├── tools/          # 22 built-in tools
│   ├── channels/       # 11 channel adapters
│   ├── skills/         # Skills system + meta orchestrator
│   ├── sandbox/        # Security sandbox (3 platforms)
│   ├── scheduler/      # Cron scheduler
│   ├── cli/            # CLI + TUI (clap + ratatui)
│   ├── safety/         # Injection detection, permissions
│   ├── mcp/            # MCP client + server
│   ├── recovery/       # Crash recovery, repair
│   ├── search/         # 7 search provider adapters
│   ├── observability/  # Tracing, logging
│   ├── eval/           # Benchmarks, scenarios
│   └── ...             # 14 more crates
├── src-tauri/          # Tauri v2 desktop shell (Rust)
├── opensquilla-webui/  # Vue 3 WebUI (639 files, 100% reusable)
└── docs/               # Documentation
```

---

## Credits

OpenSquilla is inspired by [OpenClaw](https://github.com/openclaw/openclaw). Bundled third-party content is attributed in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

Community contributors are acknowledged in [`CONTRIBUTORS.md`](CONTRIBUTORS.md).

---

## Contributing

Contributions of every kind are welcome — bug reports, feature ideas, documentation, new provider or channel adapters, skills, and core runtime work. See [`CONTRIBUTING.md`](CONTRIBUTING.md), then open an issue or pull request on [GitHub](https://github.com/opensquilla/opensquilla).

[Code of Conduct](CODE_OF_CONDUCT.md) · [Security](SECURITY.md) · [Privacy](PRIVACY.md) · [Code signing policy](docs/code-signing-policy.md) · [Third-party notices](THIRD_PARTY_NOTICES.md) · [Support](SUPPORT.md) · [License](LICENSE) (Apache-2.0)