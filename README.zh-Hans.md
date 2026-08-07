# OpenSquilla — 高效省 Token 的 AI Agent

<p align="center">
  <img src="assets/opensquilla-long-logo.png" alt="OpenSquilla logo" width="500">
</p>

<p align="center">
  <b>同样的预算，让 Agent 做更多事、做更好的事。</b><br>
  原生 Rust AI Agent —— 桌面端、CLI、聊天渠道共用一个原生二进制运行时。
</p>

<p align="center">
  <a href="https://github.com/opensquilla/opensquilla/actions/workflows/rust-ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/opensquilla/opensquilla/rust-ci.yml?style=for-the-badge" alt="CI"></a>
  <a href="https://opensquilla.ai/"><img src="https://img.shields.io/badge/website-opensquilla.ai-blue?style=for-the-badge" alt="Website"></a>
  <a href="https://github.com/opensquilla/opensquilla/releases"><img src="https://img.shields.io/github/v/release/opensquilla/opensquilla?include_prereleases&style=for-the-badge" alt="GitHub release"></a>
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/rust-stable-orange?style=for-the-badge" alt="Rust"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue?style=for-the-badge" alt="Apache 2.0 License"></a>
</p>

<p align="center">
  <a href="README.md">English</a> · <b>中文</b> · <a href="README.ja.md">日本語</a> · <a href="README.fr.md">Français</a> · <a href="README.de.md">Deutsch</a> · <a href="README.es.md">Español</a>
</p>

> 本文档与英文 [`README.md`](README.md) 同步。如有出入，请以英文版为准。

---

## 概览

OpenSquilla 是一个高效利用 Token 的 AI Agent，以 **原生 Rust 桌面应用**（Tauri v2）形式构建。本地模型路由会把每一轮都发给能处理它的最便宜模型；持久记忆、分层沙箱、内置网络搜索和设备端嵌入共同构成了一个统一共享的轮次循环。

每个入口——桌面 Web UI、CLI、TUI 和聊天渠道——都跑在同一个循环里，因此工具调度、重试和决策日志的行为处处一致。可插拔的提供商层对接 TokenRhythm、OpenRouter、OpenAI、Anthropic、Ollama、DeepSeek、Gemini、Qwen/DashScope 等 50+ 个 LLM 提供商，无需改动你的代码或配置结构。

> ⚠️ **Python 后端已完全弃用。** OpenSquilla 0.5.x 是一次完整的 Rust + Tauri v2 重写。旧的 `src/opensquilla/` Python 包、`uv`/`pip`/`wheel` 安装路径、Electron 外壳以及 `install_source` 脚本都已不再是运行时——它们仅作为遗留参考保留在仓库中，不再更新。请按下方说明构建并安装**原生 Rust 二进制**。

---

## 快速开始

### 桌面端（推荐）

从 [Releases 页面](https://github.com/opensquilla/opensquilla/releases) 下载最新安装包：

| 平台 | 格式 |
|------|------|
| Windows | `.msi` / `.exe`（NSIS） |
| macOS | `.dmg`（通用二进制 —— Apple Silicon + Intel） |
| Linux | `.deb` / `.AppImage` |

启动应用，运行 onboarding 向导，即可开始对话。桌面安装包把 Vue 3 WebUI、Agent 运行时，以及全部提供商/工具/渠道适配器都打包进单个二进制——没有 Python，没有 Electron，没有独立的 gateway 进程。

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

## 从源码构建

### 前置条件

| 要求 | 版本 |
|------|------|
| Rust | stable（见 `rust-toolchain.toml`，edition 2024） |
| Node.js | 22.12+（用于构建 WebUI） |
| npm | 10+ |

**Linux** 还需要 Tauri WebView 所需的系统库：

```sh
sudo apt-get install -y libwebkit2gtk-4.1-dev libxdo-dev libssl-dev \
  libayatana-appindicator3-dev librsvg2-dev patchelf
```

通过 NodeSource 安装 Node.js 22.12+（Debian/Ubuntu）：

```sh
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt install -y nodejs
```

### 构建

`src-tauri` 是 workspace 成员，因此 `cargo` 会把所有构建产物输出到**仓库根目录**的 `target/`（而非 `src-tauri/target/`）。

```sh
# 构建 WebUI
cd opensquilla-webui
npm ci
npm run build
cd ..

# 构建桌面应用（28 个 crate 编译进同一个二进制）
cargo tauri build

# 或仅构建 CLI / TUI
cargo build -p opensquilla_cli
cargo build -p osq_tui
```

构建产物（仓库根目录 `target/`）：

- `target/release/opensquilla`（Windows 上为 `.exe`）—— 桌面应用（Tauri v2）
- `target/release/osq` —— CLI
- `target/release/osq-tui` —— TUI

---

## 架构

```
┌──────────────────────────────────────────────────────────────────┐
│  Tauri v2 桌面外壳（Rust）                                       │
│  ├─ 窗口 / 托盘 / 深链接 / 自动更新器                              │
│  └─ Tauri commands → Rust Agent 运行时（同一进程）                │
├──────────────────────────────────────────────────────────────────┤
│  Rust Agent 运行时（28 个 workspace crate，单二进制）            │
│  ├─ Gateway（axum HTTP/WS）                                      │
│  ├─ Agent 引擎（tokio 异步状态机）                               │
│  ├─ 30+ 个 RPC 处理器                                            │
│  ├─ 50+ 个 LLM 提供商适配器（reqwest + SSE）                     │
│  ├─ 22 个内置工具（tokio::process::Command）                     │
│  ├─ 会话管理（rusqlite）+ 记忆（FTS5 + candle）                  │
│  ├─ 11 个渠道适配器（reqwest + tokio-tungstenite）               │
│  ├─ 安全沙箱（nix + windows-rs）                                 │
│  ├─ CLI / TUI（clap + ratatui）                                  │
│  └─ 技能系统 + 调度器（tokio cron）                              │
└──────────────────────────────────────────────────────────────────┘
```

### 为什么做 Rust + Tauri 重写 —— 性能收益

旧的 Python 后端（asyncio + Starlette + httpx，以 `uv`/`pip` wheel 形式打包在 Electron 外壳里）已**完全退役**。现在一切都作为单个原生二进制运行，由 28 个 Rust workspace crate 在 Tauri v2 外壳下编译而成。这次重写是由具体的性能和部署收益驱动的，而非外观：

| 维度 | 旧 Python + Electron | 原生 Rust + Tauri v2 | 效果 |
| --- | --- | --- | --- |
| **运行时模型** | Python 解释器 + pip/uv/venv + Electron V8 多进程 | 单个静态二进制，无解释器，无 sidecar | 无 GIL、无 GC 停顿、无多进程开销 |
| **并发** | `asyncio` 单线程事件循环 | `tokio` 多线程 work-stealing 运行时 | 跨核心的真正并行工具调度 |
| **轮次循环** | Gateway 进程经 IPC/HTTP 与外壳通信 | 桌面 WebUI、CLI、TUI、渠道共享同一个进程内 `TurnRunner` | 每轮零 IPC 序列化，更低延迟 |
| **内存占用** | 解释器 + 打包依赖 + Electron 渲染器 | 单进程，release profile `lto=true`、`opt-level="s"`、`strip=true` | 同一工作负载下显著更低的内存占用 |
| **启动** | 解释器初始化 + 导入解析 + Electron 启动 | 原生二进制启动 | 冷启动在几十毫秒级 |
| **安装包大小** | wheel + 运行时库 + Electron（约数百 MB） | 每平台一个 Tauri 安装包（msi/nsis/dmg/deb/AppImage） | 更小的安装包，无需打包运行时 |
| **部署** | `uv tool install` + 系统库（`libomp`、VC++ 运行库） | 运行二进制 / 运行安装包 | 无 Python 工具链，无原生依赖排障 |

由于 Agent 引擎、提供商适配器、工具、渠道、沙箱、技能、调度器、会话存储和记忆全部位于同一进程内，一轮对话绝不跨越进程边界：工具调度、重试、决策日志和会话持久化都是直接的函数调用。Tauri v2 外壳只在这个运行时之外加上窗口、系统托盘、深链接和自动更新器——没有独立的 gateway 进程，没有 Electron 主/渲染进程的分裂。

### 运行时技术栈

| 组件 | Rust crate |
|------|-----------|
| 异步运行时 | `tokio` |
| HTTP 服务器 | `axum` |
| HTTP 客户端 | `reqwest` |
| WebSocket | `tokio-tungstenite` |
| 序列化 | `serde` |
| 数据库 | `rusqlite` |
| CLI | `clap` |
| TUI | `ratatui` |
| 日志 | `tracing` |

---

## 核心功能

| 能力 | 它做什么 |
| --- | --- |
| **省 Token 的路由** | 本地 SquillaRouter（ONNX + LightGBM）按长度、语言、代码、关键词和语义嵌入给每一轮打分，再在四个分级（C0–C3）里把它分派给能胜任的最便宜模型。分类在本机上完成；做这个判断时你的提示词不会离开本机。 |
| **50+ 个 LLM 提供商** | TokenRhythm、OpenRouter、OpenAI、Anthropic、Ollama、DeepSeek、Gemini、DashScope/Qwen、Moonshot、Mistral、Groq、Zhipu、SiliconFlow、vLLM、LM Studio 等等。 |
| **自适应推理与提示** | OpenSquilla 仅对路由判定为复杂的轮次请求扩展推理，系统提示也随任务复杂度伸缩——简单轮次用轻量提示，复杂轮次用完整指令。 |
| **按需技能与 MCP** | 15+ 个内置技能（coding、GitHub、cron、pptx/docx/xlsx/pdf、摘要、天气等）仅在任务需要时加载。OpenSquilla 是 MCP 客户端，也可以作为 MCP 服务端运行。技能可从 CLI 编写、安装和发布。 |
| **持久化本地记忆** | 一份精选的 `MEMORY.md` 加上带日期的 Markdown 笔记，通过 SQLite 全文关键词搜索和 `sqlite-vec` 语义召回来检索。嵌入通过内置 ONNX 在设备端运行，也可切换到 OpenAI/Ollama。可选的指数衰减和需主动启用的“做梦（dream）”记忆整合也可用。 |
| **分层安全沙箱** | 基于权限矩阵的三档策略（Standard / Strict / Locked）。Linux 上用 Bubblewrap（及 seccomp-BPF）隔离代码执行，macOS 用 Seatbelt，Windows 用原生 `CreateRestrictedToken`。拒绝账本（denial ledger）会在反复拒绝后自动暂停自主运行，清除被拒的输出；技能元数据和工具结果也会做转义，以防提示注入。 |
| **22 个内置工具** | 文件读/写/编辑、shell 与后台进程、git、网络搜索（7 个提供商：DuckDuckGo、Bocha、Brave、Tavily、Exa 等），以及带 SSRF 防护的网页抓取、电子表格/PPTX/PDF 创作、图像生成、文本转语音。 |
| **11 个渠道** | Slack、Discord、Telegram、Feishu（飞书）、DingTalk（钉钉）、QQ、WeCom（企业微信）、Matrix、MS Teams、Terminal、WebSocket。 |
| **统一网关** | 一个运行在 `127.0.0.1:18791` 上、带 WebSocket RPC 和内嵌控制台（`/control/`）的 axum 服务。桌面 WebUI、CLI、TUI 和所有渠道共用同一个 `TurnRunner`。 |
| **持久会话、子 Agent 与调度** | 由 SQLite 支撑的会话、转录和回放存储，并带有按 Agent 隔离的工作区。Agent 可以派生深度受限的子 Agent；`SchedulerEngine` 内置了 cron 解析器，会通过 `osq cron` 运行周期性作业。 |
| **操作者控制** | 人在环路（human-in-the-loop）审批可以暂停敏感的工具调用，等人来决定；按轮次和按会话的 Token 与成本汇总（`osq cost`）及诊断信息均可从 CLI 和 Web UI 获取。 |

---

## 配置

### 首次配置

`osq onboard` 是交互式的首次配置向导。它会写入当前配置文件；当你传入 `--api-key-env` 时，提供商密钥会留在环境变量里。

```sh
osq onboard                # 完整交互式向导
osq onboard --if-needed    # 幂等：适用于脚本和重装
osq onboard --minimal      # 仅配置提供商；跳过渠道与搜索
osq onboard status         # 查看每个配置项，但不写入
```

在 SSH、CI 或任何没有 TTY 的环境中，请使用非交互形式——把密钥放在环境变量里，并传入它的**变量名**，而不是它的值：

**Linux / macOS**

```sh
export OPENROUTER_API_KEY="sk-..."
osq onboard --provider openrouter --api-key-env OPENROUTER_API_KEY
```

**Windows PowerShell**

```powershell
$env:OPENROUTER_API_KEY="sk-..."
osq onboard --provider openrouter --api-key-env OPENROUTER_API_KEY
```

OpenRouter 仅作示例——可替换为任意受支持的提供商及其对应的 API key 变量。

**配置加载顺序：** `OPENSQUILLA_GATEWAY_CONFIG_PATH` → `./opensquilla.toml` → `~/.opensquilla/config.toml` → 内置默认值。对单个密钥来说，环境变量里的值始终优先于配置文件里的值。所有选项见 `opensquilla.toml.example`。

### 运行

```sh
osq chat                       # 交互式 REPL
osq agent -m "你的提示词"        # 一次性执行，便于自动化
osq-tui                        # TUI 模式
```

其他命令组包括 `sessions`、`skills`、`memory`、`migrate`、`cron`、`channels`、`providers`、`models` 和 `cost`。运行 `osq --help` 或 `osq <组名> --help` 查看详情。

---

## 基准测试结果

PinchBench 1.2.1 在 25 个任务上的平均结果：

| Agent | 基座模型 | 平均分 | 总输入 token | 总输出 token | 总成本 |
| --- | ---: | ---: | ---: | ---: | ---: |
| OpenSquilla | 模型路由（Opus4.7、GLM5.1、DS4 Flash） | 0.9251 | 1,721,328 | 61,475 | $0.688 |
| OpenClaw | Claude Opus 4.7 | 0.9255 | 3,066,243 | 50,890 | $6.233 |

分数是 25 个任务的均值；token 数和成本是整次运行的总计。

---

## 项目结构

```
├── crates/             # 28 个 Rust workspace crate
│   ├── engine/         # Agent 运行时、轮次运行器、路由
│   ├── gateway/        # axum HTTP/WS 服务器，30+ 个 RPC 处理器
│   ├── provider/       # 50+ 个 LLM 提供商适配器
│   ├── session/        # 会话管理（rusqlite）
│   ├── memory/         # 记忆系统（FTS5 + sqlite-vec）
│   ├── tools/          # 22 个内置工具
│   ├── channels/       # 11 个渠道适配器
│   ├── skills/         # 技能系统 + meta 编排器
│   ├── sandbox/        # 安全沙箱（3 个平台）
│   ├── scheduler/      # cron 调度器
│   ├── cli/            # CLI + TUI（clap + ratatui）
│   ├── safety/         # 注入检测、权限
│   ├── mcp/            # MCP 客户端 + 服务端
│   ├── recovery/       # 崩溃恢复、修复
│   ├── search/         # 7 个搜索提供商适配器
│   ├── observability/  # tracing、日志
│   ├── eval/           # 基准测试、场景
│   └── ...             # 另外 13 个 crate（core、onboarding、identity、
│                       #   persistence、health、plugins、uninstall、
│                       #   chat、contrib、agents 等）
├── src-tauri/          # Tauri v2 桌面外壳（workspace 成员）
├── opensquilla-webui/  # Vue 3 WebUI（688 个文件，内嵌进二进制）
├── src/opensquilla/    # ⚠️ 旧 Python 后端 —— 已弃用，非运行时
└── docs/               # 文档
```

---

## 致谢

OpenSquilla 的灵感来自 [OpenClaw](https://github.com/openclaw/openclaw)。内置的第三方内容在 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) 中注明出处。

社区贡献者都记在 [`CONTRIBUTORS.md`](CONTRIBUTORS.md) 里。

---

## 参与贡献

我们欢迎各种形式的贡献——bug 报告、功能想法、文档、新的提供商或渠道适配器、技能，以及核心运行时方面的开发。请参阅 [`CONTRIBUTING.md`](CONTRIBUTING.md)，然后到 [GitHub](https://github.com/opensquilla/opensquilla) 上提 issue 或 pull request。

[行为准则](CODE_OF_CONDUCT.md) · [安全](SECURITY.md) · [隐私](PRIVACY.md) · [代码签名策略](docs/code-signing-policy.md) · [第三方声明](THIRD_PARTY_NOTICES.md) · [支持](SUPPORT.md) · [许可证](LICENSE)（Apache-2.0）
