# OpenSquilla Python 后端功能清单 & Tauri 迁移可行性映射

> 生成日期: 2026-08-05
> 总代码量: ~960 个 Python 文件, ~292,000 行 (含所有子系统)

---

## 一、整体架构概览

```
┌──────────────────────────────────────────────────────────────────┐
│  Tauri 桌面壳 (Rust) — 替换 Electron                             │
│  ├─ HTTP/WS 网关 (axum/tokio-tungstenite 替换 Starlette)          │
│  ├─ 窗口管理 / 系统托盘 / 深层链接 / 安全存储                     │
│  └─ Agent 运行时 (Tokio 异步运行时)                                │
├──────────────────────────────────────────────────────────────────┤
│  Rust 核心 Agent 运行时 (全部 Python 后端重写)                      │
│  ├─ Gateway HTTP/WS 服务 (axum)                                    │
│  ├─ Agent 引擎 (Tokio async state machine)                         │
│  ├─ 30+ RPC 处理器                                                 │
│  ├─ 50+ LLM 提供商适配器 (reqwest + SSE)                           │
│  ├─ 22 个内置工具 (tokio::process::Command)                        │
│  ├─ 会话管理 (rusqlite) + 记忆系统 (FTS5 + candle)                 │
│  ├─ 多信道消息 (reqwest + tokio-tungstenite)                       │
│  ├─ 安全沙箱 (nix + windows-rs)                                    │
│  ├─ 命令行/TUI (clap + ratatui)                                   │
│  └─ 技能系统 + 定时任务调度器 (tokio cron)                         │
└──────────────────────────────────────────────────────────────────┘
```

---

## 二、子系统详细清单

### 2.1 Gateway 网关 (64,833 行, 80+ 文件)

#### 核心服务

| 文件 | 行数 | 功能 | Tauri 迁移策略 | 状态 |
|------|------|------|---------------|------|
| `app.py` | 831 | Starlette ASGI 应用工厂, 路由注册 | ✅ `crates/gateway/src/app.rs` — axum 实现 (339 行) | ✅ |
| `boot.py` | 4,568 | 启动编排: 构建所有服务, DB 迁移, 优雅关闭 | ✅ `crates/gateway/src/boot.rs` — tower::ServiceBuilder 编排 (495 行) | ✅ |
| `config.py` | 3,353 | Pydantic v2 配置管理, 30+ 子配置, TOML 加载, 配置迁移 | ✅ `crates/gateway/src/config.rs` — serde + toml (873 行) | ✅ |
| `websocket.py` | 1,184 | WebSocket 连接处理, 挑战-响应认证, 帧解析, 订阅管理 | ✅ `crates/gateway/src/websocket.rs` — axum WS (850 行) | ✅ |
| `control_ui.py` | 321 | Vue.js SPA 静态文件服务 + Jinja2 模板渲染 | ✅ Tauri 内置资产服务 | ✅ |
| `protocol.py` | 210 | WebSocket 协议帧类型 (Pydantic 模型) | ✅ `crates/gateway/src/protocol.rs` — serde struct (860 行) | ✅ |
| `auth.py` | 199 | 服务端认证: Token 认证 / 开放认证 + 回环升级 | ✅ `crates/gateway/src/auth.rs` (888 行) | ✅ |
| `middleware.py` | 317 | 中间件: Origin 检查, 认证, 限流, 安全头, 错误处理 | ✅ `crates/gateway/src/middleware.rs` (778 行) | ✅ |

#### RPC 处理器 (30 个文件)

| 文件 | 行数 | 功能 | 迁移策略 |
|------|------|------|---------|
| `rpc_sessions.py` | ~1,200 | 会话生命周期, Turn 管理, Compaction, 附件 | ✅ `crates/gateway/src/sessions.rs` (1,463 行) |
| `rpc_chat.py` | ~600 | 聊天发送, 历史, 附件上传/删除 | ✅ `crates/gateway/src/chat.rs` (911 行) |
| `rpc_config.py` | ~500 | 配置 CRUD, 模型路由 | ✅ `crates/gateway/src/config.rs` (873 行) |
| `rpc_memory.py` | ~500 | 记忆检查, 搜索, 修复, 刷新 | ✅ `crates/gateway/src/memory.rs` (324 行) |
| `rpc_memory_import.py` | ~500 | AI 辅助配置导入 | ✅ `crates/gateway/src/memory_import.rs` |
| `rpc_onboarding.py` | ~500 | 提供商/信道配置变更 | ✅ `crates/gateway/src/onboarding.rs` (335 行) |
| `rpc_sandbox.py` | ~500 | 沙箱运行上下文管理 | ✅ `crates/gateway/src/sandbox.rs` (358 行) |
| `rpc_skills.py` | ~400 | 技能目录, 安装, 启用/禁用 | ✅ `crates/gateway/src/skills.rs` (409 行) |
| `rpc_cron.py` | ~400 | 定时任务管理 | ✅ `crates/gateway/src/cron.rs` (432 行) |
| `rpc_doctor.py` | ~400 | 统一健康检查 | ✅ `crates/gateway/src/doctor.rs` (231 行) |
| `rpc_meta_runs.py` | ~400 | Meta-skill 运行历史 | ✅ `crates/gateway/src/meta_runs.rs` |
| `rpc_channels.py` | ~400 | 信道管理 | ✅ `crates/gateway/src/channels.rs` (410 行) |
| `rpc_approvals.py` | ~300 | 审批队列 | ✅ `crates/gateway/src/approvals.rs` (366 行) |
| `rpc_migration.py` | ~300 | 配置发现/预览 | ✅ `crates/gateway/src/migration.rs` |
| `rpc_usage.py` | ~300 | 用量跟踪和成本 | ✅ `crates/gateway/src/usage.rs` (348 行) |
| `rpc_agents.py` | ~230 | Agent CRUD, 工作区文件检查 | ✅ `crates/gateway/src/agents.rs` (340 行) |
| `rpc_commands.py` | ~200 | 斜杠命令目录 | ✅ `crates/gateway/src/commands.rs` (382 行) |
| `rpc_logs.py` | ~200 | 日志检查 | ✅ `crates/gateway/src/logs.rs` |
| `rpc_routing.py` | ~200 | 每会话路由保持 | ✅ `crates/gateway/src/routing.rs` |
| `rpc_router.py` | ~200 | 路由决策记录 | ✅ `crates/gateway/src/router.rs` |
| `rpc_models.py` | ~200 | 模型目录列表 | ✅ `crates/gateway/src/models.rs` (252 行) |
| `rpc_proposals.py` | ~100 | Meta-skill 提案 | ✅ `crates/gateway/src/proposals.rs` |
| `rpc_wizard.py` | ~90 | 引导向导状态机 | ✅ `crates/gateway/src/wizard.rs` |
| `rpc_diagnostics.py` | ~40 | 诊断开关 | ✅ `crates/gateway/src/diagnostics.rs` |
| `rpc_secrets.py` | ~30 | 密钥管理 (存根) | ✅ `crates/gateway/src/secrets.rs` |
| `rpc_tools.py` | ~200 | 工具目录, 搜索, 提供商状态 | ✅ `crates/gateway/src/tools.rs` (442 行) |
| `rpc_system.py` | ~200 | 系统/消息 RPC | ✅ `crates/gateway/src/system.rs` |
| `rpc_workspaces.py` | ~200 | 项目工作区生命周期 | ✅ `crates/gateway/src/workspaces.rs` (355 行) |

#### 其他网关模块

| 文件 | 功能 | 策略 |
|------|------|------|
| `session_events.py` | 会话事件广播 | **Rust 重写**. tokio::broadcast |
| `session_lifecycle.py` | 会话生命周期 | **Rust 重写**. 状态机 |
| `session_services.py` | 会话服务 | **Rust 重写**. 依赖注入 |
| `session_streams.py` | 会话流 | **Rust 重写**. tokio Stream |
| `turn_ingress.py` | 入站 Turn 处理 | **Rust 重写**. 消息入站管道 |
| `attachments.py` | 附件管理 | **Rust 重写**. 文件操作 |
| `uploads.py` | 文件上传 | **Rust 重写**. axum multipart |
| `artifact_preview.py` | Artifact 预览 | **Rust 重写** |
| `audio_transcription.py` | 音频转写 | **Rust 重写**. HTTP 转写 API |
| `desktop_ownership.py` | 桌面所有权验证 | **Rust 重写** |
| `pidlock.py` | 进程 PID 锁 | **Rust 重写** |
| `model_routing.py` | 模型路由 | **Rust 重写** |
| `provider_stats.py` | 提供商统计 | **Rust 重写** |
| `routing.py` | 请求路由 | **Rust 重写** |
| `scopes.py` | 作用域定义 | **Rust 重写** |
| `static/` | 静态文件 | **Rust 重写**. Tauri 内置资产服务 |
| `templates/` | Jinja2 模板 | **Rust 重写**. Tauri 前端渲染 |
| `rpc/registry.py` | RPC 分发注册表 | **Rust 重写** |

---

### 2.2 Provider 提供商引擎 (34,418 行, 49+ 文件)

#### 提供商适配器 (5 个后端, 50+ 注册提供商)

| 后端类型 | 提供商列表 | 适配器文件 | 行数 | 状态 |
|---------|-----------|-----------|------|------|
| `openai_compat` | openai, deepseek, gemini, dashscope, qwen, moonshot, mistral, groq, zhipu, siliconflow, openrouter, azure, 等 30+ | `openai.py` | 254,692 | ✅ `crates/provider/src/openai.rs` — 54 OpenAIProvider variants (3,438 行) |
| `anthropic` | anthropic, minimax, qwen_token_plan_anthropic, 等 10+ | `anthropic.py` | 52,332 | ✅ `crates/provider/src/anthropic.rs` (1,705 行) |
| `openai_responses` | openai_responses, volcengine_coding_plan, byteplus_coding_plan | `openai_responses.py` | 44,203 | ✅ `crates/provider/src/openai_responses.rs` (2,469 行) |
| `openai_codex` | openai_codex | `openai_codex.py` | 39,272 | ✅ `crates/provider/src/openai_codex.rs` (1,393 行) |
| `ollama` | ollama | `ollama.py` | 34,348 | ✅ `crates/provider/src/ollama.rs` (1,144 行) |
| `ensemble` | 多模型集成 (proposer-aggregator) | `ensemble.py` | 171,640 | ✅ `crates/provider/src/ensemble.rs` (3,463 行) |

**关键发现: 所有提供商都使用原始 HTTP (httpx), 没有任何 Python SDK 依赖.**

- 没有 `openai` pip 包
- 没有 `anthropic` pip 包
- 没有 `google-generativeai`

#### 核心基础设施

| 文件 | 行数 | 功能 | 迁移策略 |
|------|------|------|---------|
| `registry.py` | 676 | 提供商注册表 (50+ 声明式 ProviderSpec) | ✅ `crates/provider/src/registry.rs` (628 行) |
| `selector.py` | 742 | 模型选择器 + 回退链 | ✅ `crates/provider/src/selector.rs` (258 行) |
| `protocol.py` | 368 | LLMProvider 协议接口 | ✅ `crates/provider/src/types.rs` (ChatProvider trait) |
| `types.py` | 657 | 流事件, ChatConfig, Message, ModelInfo | ✅ `crates/provider/src/types.rs` (191 行) + `crates/provider/src/stream.rs` |
| `stream_assembly.py` | 30,937 | 推理/工具调用流组装 | ✅ `crates/provider/src/stream_assembly.rs` (726 行) |
| `text_tool_normalizer.py` | 95,170 | 文本到工具方言规范化 | ✅ `crates/provider/src/text_tool_normalizer.rs` (743 行) + `crates/provider/src/normalizer.rs` (321 行) |
| `request_proof.py` | 89,921 | 预检载荷投影/预算 | ✅ `crates/provider/src/request_proof.rs` (686 行) |
| `failures.py` | 16,774 | 错误分类 + 恢复动作 | ✅ `crates/provider/src/failures.rs` (589 行) |
| `compat_policy.py` | 21,286 | OpenAI 兼容性策略方言数据 | ✅ `crates/provider/src/compat_policy.rs` (812 行) |
| `model_catalog.py` | 66,522 | 分层模型元数据缓存 | ✅ `crates/provider/src/model_catalog.rs` (416 行) + `crates/provider/src/live_catalog.rs` (394 行) |
| `live_catalog.py` | 8,433 | 启动时实时目录抓取 | ✅ 同上 |
| `credentials.py` | 157 | 凭据池 (轮询 + 429 冷却) | ✅ `crates/provider/src/credentials.rs` (362 行) |
| `image_generation.py` | 55,679 | 图像生成适配器 | ✅ `crates/provider/src/image_generation.rs` (3,007 行) |
| `audio.py` | 26,961 | 音频 TTS (ElevenLabs) | ✅ `crates/provider/src/audio.rs` (2,245 行) |

#### 模型路由 (squilla_router, 5,929 行)

| 组件 | 功能 | 迁移策略 |
|------|------|---------|
| V4 Phase 3 推理引擎 | ONNX 模型推理 (BGE 编码器 + 分类器) | **Rust 原生**. `ort` crate 替代 onnxruntime, `candle` 替代 numpy, `tokenizers` crate 替代 HuggingFace tokenizers |
| Controller (`controller.py`) | 纯函数: 思维模式推导, 提示策略 | **Rust 重写** |
| Self-learning (`self_learning/`) | 反馈捕获, 数据集构建, 模型训练 | **Rust 重写**. 反馈收集 + 数据集导出, 外部训练脚本 |

---

### 2.3 Engine 引擎 (59,126 行, 50+ 文件)

> 核心 Agent 循环, 全部用 Rust 重写

| 文件/目录 | 行数 | 功能 | 迁移策略 |
|-----------|------|------|---------|
| `runtime.py` | 10,441 | 主 Agent 运行时: `TurnRunner` 类, 编排 8 个 stage, 会话生命周期, 用量会计 | ✅ `crates/engine/src/runtime.rs` (1,284 行) |
| `agent.py` | 19,290 | Agent 状态机 + 工具循环: `_turn_generator()` ~3000 行显式状态机, `provider.chat()` LLM 调用, 13 处 `subprocess.run()` git 操作 | ✅ `crates/engine/src/agent.rs` (2,449 行) |
| `context.py` | 72 | 上下文组装: 加载 SOUL.md/AGENTS.md 等工作区文件 | ✅ `crates/engine/src/context.rs` (234 行) |
| `turn_control.py` | 145 | 纯决策逻辑: `decide_turn_control()` 分类停止表面, 无 I/O | ✅ `crates/engine/src/turn_control.rs` (319 行) |
| `pipeline.py` | 129 | 预 Turn 管道: `for step in steps: ctx = await step(ctx)` 失败-开放语义 | ✅ `crates/engine/src/pipeline.rs` (225 行) + `crates/engine/src/steps/` (5 个 steps) |
| `history.py` | 374 | 对话历史: 截断, 工具配对修复, 持久化条目重建 | ✅ `crates/engine/src/history.rs` (443 行) |
| `commands.py` | 714 | 斜杠命令注册表: Surface, ExecutionKind, CommandCategory 枚举 | ✅ `crates/engine/src/commands.rs` (337 行) |
| `start_turn.py` | ~100 | Turn 入队: `reserve_turn_via_runtime()`, `start_turn_via_runtime()` | ✅ 网关耦合, 重新设计 (runtime.rs 中) |
| `thinking.py` | 98 | 推理清理: `drop_reasoning()` 跨提供商兼容性 | ✅ `crates/engine/src/thinking.rs` (191 行) |
| `subagent.py` | 722 | 子 Agent 管理: 进程内 Agent 实例, 提供商克隆 | ✅ `crates/engine/src/subagent.rs` (298 行) |
| `usage.py` | 825 | Token 用量跟踪: `UsageTracker`, `ContextVar` 作用域绑定 | ✅ `crates/engine/src/usage.rs` (271 行) |
| `usage_accounting.py` | 803 | 提供商调用用量会计: `UsageEventSink` 协议 | ✅ `crates/engine/src/usage.rs` |
| `pricing.py` | 701 | 模型定价: `PricingCache`, OpenRouter 1h TTL 缓存 | ✅ `crates/engine/src/pricing.rs` (728 行) |
| `routing/` | ~2,000 | 路由策略引擎: `RoutingPolicyEngine`, 校准文件, 健康账本 | ✅ `crates/engine/src/routing/` (4 文件, 3,005 行) |
| `steps/` | ~4,000 | 预 Turn 管道步骤: squilla_router(ML), meta_resolution, model_select, skills_filter 等 | ✅ `crates/engine/src/steps/` (5 文件, 1,441 行) |
| `turn_runner/` | ~8,000 | 8 个 Stage 类: harness, agent_bootstrap, attachment, compaction, input, provider, stream_consumer, finalizer | ✅ `crates/engine/src/turn_runner/` (9 文件, 5,602 行) |
| `hooks/` | ~300 | 钩子协议: TurnHook, CompactionHook, ToolHook | ✅ `crates/engine/src/hooks.rs` (274 行) |
| `runtime_recovery.py` | ~200 | 空/无进展 Turn 恢复: 纯决策函数 | ✅ `crates/engine/src/runtime_recovery.rs` (295 行) |
| `session_lock.py` | 49 | 每会话异步锁 (`asyncio.Lock` per key) | ✅ `crates/engine/src/session_lock.rs` (317 行) |
| `compaction_control.py` | 91 | 压缩延续决策: 纯函数 | ✅ `crates/engine/src/compaction_control.rs` (254 行) |

---

### 2.4 Tools 工具系统 (33,060 行, 30+ 文件)

#### 22 个内置工具

| 工具 | 文件 | 功能 | 核心依赖 | OS 操作 | 状态 |
|------|------|------|---------|---------|------|
| `exec_command` | `shell.py` (5,100 行) | 任意 Shell 命令执行 | subprocess | **子进程** | ✅ `crates/tools/src/shell.rs` (698 行) |
| `background_process` | `shell.py` | 后台进程管理 | subprocess | **子进程** | ✅ `crates/tools/src/shell.rs` |
| `execute_code` | `code_exec.py` | Python 代码执行 | subprocess | **子进程** | ✅ `crates/tools/src/code_exec.rs` (452 行) |
| `read_file` | `filesystem.py` | 读文件 | - | **文件系统** | ✅ `crates/tools/src/filesystem.rs` (388 行) |
| `write_file` | `filesystem.py` | 写文件 | - | **文件系统** | ✅ 同上 |
| `edit_file` | `filesystem.py` | 编辑文件 | - | **文件系统** | ✅ 同上 |
| `apply_patch` | `patch.py` | unified-diff 补丁 | subprocess | **子进程** | ✅ `crates/tools/src/patch.rs` (405 行) |
| `web_search` | `web.py` | 搜索引擎 API | httpx | **网络** | ✅ `crates/tools/src/web.rs` (528 行) |
| `web_fetch` | `web_fetch.py` | URL 内容提取 | httpx, readability-lxml | **网络** | ✅ 同上 |
| `http_request` | `web.py` | 任意 HTTP 请求 | httpx | **网络** | ✅ 同上 |
| `git_*` | `git.py` | Git 操作 | subprocess | **子进程 + 文件系统** | ✅ `crates/tools/src/git.rs` (313 行) |
| `image` | `media.py` | 图片处理 | Pillow, httpx | **文件系统 + 网络** | ✅ `crates/tools/src/media.rs` (601 行) |
| `pdf` | `media.py` | PDF 读取 | pdfplumber | **文件系统** | ✅ 同上 |
| `tts` | `media.py` | 文字转语音 | httpx | **网络** | ✅ 同上 |
| `generate_*` | `file_authoring.py` | 文件生成 | fpdf2, openpyxl | **文件系统** | ✅ `crates/tools/src/file_authoring.rs` (1,031 行) |
| `memory_*` | `memory_tools.py` | 记忆操作 | sqlite3 | **文件系统** | ✅ `crates/tools/src/memory_tools.rs` (516 行) |
| `session_*` | `sessions.py` | 会话管理 | - | **文件系统** | ✅ `crates/tools/src/session_tools.rs` (715 行) |
| `message` | `messaging.py` | 信道消息 | - | **网络** | ✅ `crates/tools/src/messaging.rs` (139 行) |
| `cron` | `admin.py` | 定时任务 | - | **文件系统** | ✅ `crates/tools/src/cron_tool.rs` (474 行) |
| `publish_artifact` | `artifacts.py` | Artifact 生成 | - | **文件系统** | ✅ `crates/tools/src/artifacts.rs` (542 行) |

#### 工具基础设施

| 模块 | 功能 | 迁移策略 |
|------|------|---------|
| `dispatch.py` (1,700 行) | 中央工具分发: 注入防护, 策略链, 预算, 沙箱 | ✅ `crates/tools/src/dispatch.rs` (610 行) + `crates/tools/src/policy.rs` (608 行) |
| `registry.py` | 工具注册表 + @tool 装饰器 | ✅ `crates/tools/src/registry.rs` (814 行) |
| `policy/` | 工具策略链 (deny, budget, finalize) | ✅ `crates/tools/src/policy.rs` |
| `ssrf.py` | SSRF 保护 (DNS 解析, IP 范围检查) | ✅ `crates/tools/src/ssrf.rs` (412 行) |
| `sandbox.py` | 沙箱集成 | ✅ `crates/sandbox/src/` (9 文件, 1,546 行) 🔄 正在扩展 |
| `schema_validation.py` | JSON Schema 参数验证 | ✅ `crates/tools/src/schema_validation.rs` (241 行) |

#### MCP 生态

| 模块 | 行数 | 功能 | 迁移策略 |
|------|------|------|---------|
| **Outbound MCP** (`mcp/`) | 670 | 连接外部 MCP 服务器, 导入工具 | ✅ `crates/mcp/src/` (5 文件, 1,939 行) |
| **Inbound MCP** (`mcp_server/`) | 399 | 对外暴露会话操作 | ✅ 同上 |

---

### 2.5 Session 会话管理 (19,228 行, 20 文件)

#### 核心组件

| 文件 | 行数 | 功能 | 迁移策略 |
|------|------|------|---------|
| `storage.py` | ~2,700 | 底层异步 SQLite CRUD | ✅ `crates/session/src/storage.rs` (2,671 行) |
| `manager.py` | ~2,730 | 高层生命周期: 创建/恢复/分支/Fork/Kill/Compaction | ✅ `crates/session/src/manager.rs` (1,543 行) |
| `models.py` | 412 | SQLModel ORM 模型 (19 个表) | ✅ 内置在 storage.rs |
| `keys.py` | 198 | 会话 Key 构建 | ✅ `crates/session/src/naming.rs` (503 行) |
| `compaction.py` | ~900 | 上下文窗口压缩 (LLM 摘要) | ✅ `crates/session/src/compaction.rs` (1,106 行) |
| `naming.py` | ~550 | 自动命名会话 (LLM) | ✅ `crates/session/src/naming.rs` 包含 |
| `plans.py` | ~500 | 协作计划状态机 | ✅ `crates/session/src/plans.rs` (966 行) |
| `usage_ledger.py` | ~500 | 用量账本 (纳美元精度) | ✅ 内置在 manager.rs |
| `material_cleanup.py` | ~80 | 文件系统清理钩子 | ✅ 内置在 manager.rs |

#### 数据库设计

**数据库: SQLite 独占** (aiosqlite 异步包装器)

**会话 DB 表** (19 个表):
- `sessions` — 会话元数据, 路由, Token 跟踪, 成本
- `transcript_entries` — 消息记录 (角色, 内容, 工具调用, 推理)
- `compacted_transcript_entries` — Compaction 移出的行
- `session_summaries` — Compaction 摘要记录
- `session_context_states` — 上下文状态
- `plan_revisions`, `plan_runs` — 协作计划
- `agent_tasks` — 任务运行时账本
- `project_workspaces` — 项目目录
- `usage_events`, `usage_event_items`, `usage_ledger_state` — 用量
- 等 19 个表

**记忆 DB 表** (6 个表):
- `files` — 索引文件跟踪
- `chunks` — 文本块 + 嵌入向量
- `chunks_fts` — FTS5 全文搜索虚拟表
- `chunks_vec` — sqlite-vec 向量搜索虚拟表 (可选)
- `embedding_cache` — 嵌入缓存
- `meta` — 元数据

---

### 2.6 Memory 记忆系统 (15,856 行, 42 文件)

| 组件 | 行数 | 功能 | 迁移策略 |
|------|------|------|---------|
| `store.py` | ~1,370 | 核心 SQLite 记忆存储 (FTS5 + sqlite-vec) | ✅ `crates/memory/src/store.rs` (1,246 行) |
| `embedding.py` | ~550 | 嵌入提供者 (OpenAI/Ollama/ONNX) | ✅ `crates/memory/src/embedding.rs` |
| `retrieval.py` | ~280 | 混合搜索 (向量 + BM25 + 时间衰减 + MMR) | ✅ `crates/memory/src/retrieval.rs` |
| `manager.py` | ~660 | 每 Agent 记忆管理器 | ✅ `crates/memory/src/manager.rs` |
| `sync_manager.py` | ~300 | 统一同步触发 (文件监视器, 定时器, TTL) | ✅ `crates/memory/src/sync.rs` (700 行) |
| `turn_capture.py` | ~200 | Turn 增量持久化 | ✅ `crates/memory/src/turn_capture.rs` (871 行) |
| `session_source.py` | ~200 | 会话派生记忆文档 | ✅ `crates/memory/src/session_source.rs` (1,137 行) |
| `dream/` | ~1,500 | 梦境记忆整合 (LLM 生成的记忆合并) | ✅ `crates/memory/src/dream.rs` (1,007 行) |
| `profile_import/` | ~800 | 外部配置导入记忆 | ✅ `crates/memory/src/profile_import.rs` (1,150 行) |

#### Python 特定 ML 库 vs Rust 等价物

| Python 库 | 用途 | Rust 替代 |
|-----------|------|-----------|
| `tiktoken` | Token 估算 | `tiktoken-rs` |
| `jieba` | 中文分词 | `jieba-rs` |
| `sqlite-vec` | 向量搜索 | 相同 C 扩展 |
| `onnxruntime` | 本地嵌入推理 | `ort` crate |
| `tokenizers` (HF) | ONNX 分词 | 原生 Rust |
| `httpx` | HTTP 调用 | `reqwest` |
| `numpy` | 向量运算 | `ndarray` |

---

### 2.7 Channels 多信道消息 (15,718 行, 30+ 文件)

#### 支持的信道

| 信道 | 模块 | 能力等级 | 入站机制 | 出站 | 依赖 | 迁移策略 |
|------|------|---------|---------|------|------|---------|
| **Slack** | `slack.py` | YELLOW | Webhook (Events API)/Socket Mode | `httpx` REST | 无 SDK | ✅ `crates/channels/src/slack.rs` (137 行) |
| **Discord** | `discord.py` | YELLOW | 持久化 WebSocket (Gateway API) | `httpx` REST | `websockets` | ✅ `crates/channels/src/discord.rs` (140 行) |
| **Telegram** | `telegram.py` | YELLOW | 长轮询/Webhook | `httpx` Bot API | 无 SDK | ✅ `crates/channels/src/telegram.rs` (131 行) |
| **Feishu/Lark** | `feishu.py` | YELLOW | Webhook/持久化 WS | `httpx` REST | `lark-oapi>=1.5.3` | ✅ `crates/channels/src/feishu.rs` (732 行) |
| **DingTalk** | `dingtalk.py` | YELLOW | 持久化 WS (Stream Mode) | SDK | `dingtalk-stream` | ✅ `crates/channels/src/dingtalk.rs` (871 行) |
| **QQ Bot** | `qq.py` | YELLOW | 持久化 WS | SDK REST | `qq-botpy` | ✅ `crates/channels/src/qq.rs` (845 行) |
| **WeCom** | `wecom.py` | YELLOW | Webhook/WS | `httpx` REST | 无 SDK (自含加密) | ✅ `crates/channels/src/wecom.rs` (79 行) |
| **Matrix** | `matrix.py` | YELLOW | HTTP 长轮询 | SDK | `matrix-nio` | ✅ `crates/channels/src/matrix.rs` (722 行) |
| **MS Teams** | `msteams.py` | GREEN | Bot Framework Webhook | SDK | `botbuilder` | ✅ `crates/channels/src/msteams.rs` (860 行) |
| **Terminal** | `terminal.py` | 内置 | 同步 stdin | stdout | 无 | ✅ `crates/channels/src/terminal.rs` (84 行) |
| **WebSocket** | `websocket.py` | 内置 | `asyncio.Queue` | WS 事件 | 无 | ✅ `crates/channels/src/websocket.rs` (140 行) |

#### 调度架构

```
Boot → ChannelManager.from_config()
  → 对每个信道: build_managed_channel() → ManagedChannel
  → install_outbox() (持久化出站包装)
  → register_tool_channel() (注册为消息工具)
  → collect_webhook_routes() → Starlette Route
  → manager.start_all()
    → 每个适配器: adapter.start() → 连接到提供商
    → asyncio.Task for _dispatch_with_retry()
      → run_channel_dispatch() 循环
        → channel.receive() → IncomingMessage
        → decide_channel_admission() (准入策略)
        → 斜杠命令: 通过 RPC 分发
        → Agent turn: start_turn_via_runtime()
        → 回复投递: _deliver_runtime_channel_reply()
```

#### 关键发现

- **核心信道** (Slack/Telegram/WeCom) 使用零 SDK 依赖——纯 `httpx` REST API
- **SDK 依赖信道** (Feishu/DingTalk/QQ/Matrix) 在 optional extras 后延迟加载
- 所有信道都是 `asyncio.Task` 在主事件循环中运行
- `delivery_store` 使用 SQLite 传输租约 + fencing token, 已为多进程设计
- Webhook 路径 (Slack/Feishu/Telegram/WeCom/Teams) 是 Starlette Route, 需要 Tauri 中对应的 HTTP 路由
- 短期: 作为 sidecar 进程保留 Python 适配器. 长期: 在 Rust 中重写核心适配器

---

### 2.8 Sandbox 沙箱 (29,178 行, 55+ 文件)

#### 三大平台后端

| 平台 | 后端文件 | 隔离机制 | 实现方式 |
|------|---------|---------|---------|
| **Linux** | `backend/bubblewrap.py` | `bwrap` (用户命名空间) | `--unshare-user --unshare-pid --unshare-net`, tmpfs root, ro-bind 主机路径, `--cap-drop ALL`, seccomp-BPF (ctypes), `resource.setrlimit` |
| **macOS** | `backend/seatbelt.py` (~1,390 行) | `sandbox-exec` (Seatbelt) | 生成 SBPL 配置文件, deny-by-default 文件系统和网络, 显式 allow 规则 |
| **Windows** | `backend/windows_default.py` | Windows 原生机制 | `CreateRestrictedToken`, `CreateProcessAsUser` (离线沙箱用户), ACL 文件系统控制, Windows Firewall + WFP, Job Object |
| **Noop** | `backend/noop.py` | 无隔离 | `asyncio.create_subprocess_exec` + `resource.setrlimit` |

#### 关键特性

- **子进程隔离**: 整个沙箱模型基于子进程执行 (bwrap/sandbox-exec/CreateProcessAsUser)
- **文件系统控制**: Linux (bind mount + 空 tmpfs 删除路径), macOS (SBPL 文件读写规则), Windows (ACL + 受限令牌)
- **网络控制**: 3 种模式: `NONE` (阻断), `PROXY_ALLOWLIST` (本地 HTTP 代理 + 域名白名单), `HOST` (直连)
- **策略引擎**: `policy.py` 纯函数: `select_level()` + `build_policy()`; 3 级安全级别 (STANDARD/STRICT/LOCKED)
- **审批门**: `governance.py` 审批队列 + 拒绝账本 + 拒绝后守卫
- **陈旧输出缓存**: 防止已沙箱化的操作重新执行

#### 迁移策略

**全部 Rust 重写.** Tauri 的 Rust 原生能力使沙箱实现更直接:

| 平台 | Rust 实现策略 | 关键 crate |
|------|-------------|-----------|
| **Linux** | ✅ `crates/sandbox/src/linux.rs` (961 行) | `nix`, `seccompiler` |
| **macOS** | ✅ `crates/sandbox/src/macos.rs` (496 行) | `serde` 序列化 SBPL |
| **Windows** | ✅ `crates/sandbox/src/windows.rs` (735 行) | `windows` |
| **网络代理** | ✅ `crates/sandbox/src/network.rs` (686 行) | `tokio`, `trust-dns` |
| **策略引擎** | ✅ `crates/sandbox/src/policy.rs` (745 行) | 纯 Rust 枚举 + 模式匹配 |
| **审批门** | ✅ `crates/sandbox/src/governance.rs` (583 行) | `tokio::sync::mpsc` |
| **陈旧输出缓存** | ✅ `crates/sandbox/src/stale_output_cache.rs` (447 行) | `tokio::fs` 文件哈希 |

---

### 2.9 CLI 命令行 (35,759 行, 60+ 文件)

#### 框架

| 组件 | 框架 | 说明 |
|------|------|------|
| CLI 框架 | ✅ `clap` (Rust) — `crates/cli/src/commands.rs` (266 行) | 替代 typer 的所有子命令注册 |
| TUI 渲染 | ✅ `ratatui` (Rust) — `crates/cli/src/tui.rs` (520 行) | 替代 Rich 原生 TUI + OpenTUI JS sidecar |
| OpenTUI | ✅ 不再需要 | Tauri 原生窗口替代 Bun/JS sidecar |

#### CLI 架构模式

| 模式 | 描述 | 命令示例 |
|------|------|---------|
| **A: Tauri 命令** | 通过 `tauri::command` 或 axum 内部 RPC 调用 | `sessions`, `models`, `skills`, `cost`, `channels status`, `memory` |
| **B: 直接内部模块** | 直接 `use opensquilla_core::*` 调用内部模块 | `config`, `providers list`, `onboard`, `sandbox`, `doctor`, `router calibrate`, `init` |
| **C: 网关生命周期** | 管理 Tauri 应用进程本身 | `gateway run/start/stop/status/restart` |

#### 关键命令的实现方式

| 命令 | 需要网关? | 实现方式 |
|------|----------|---------|
| `config get/set` | 否 | 直接 `Config::load()` (serde + toml) |
| `providers list` | 否 | `registry::ProviderSpec` 注册表 |
| `providers status` | 是 | 内部 RPC 调用 |
| `chat` (默认) | 是 | 内部 RPC 调用 |
| `chat --standalone` | 否 | 直接 `TurnRunner` |
| `agent` | 否 | 直接 `TurnRunner` |
| `onboard` | 否 | `onboarding::flow` 向导 |
| `doctor` | 否 | `health::check` 诊断 |
| `sandbox` | 否 | `sandbox::Sandbox` 直接调用 |
| `router calibrate` | 否 | `routing::calibration` 校准 |
| `memory dream` | 否 | `memory::dream` 梦境合并 |
| `sessions list` | 是 | 内部 RPC 调用 |
| `models list` | 是 | 内部 RPC 调用 |

#### 迁移策略

- **CLI 完全用 Rust 重写** (clap + ratatui)
- **Tauri 应用本身就是网关**: 不需要 WebSocket 通信, 所有模块在同一进程内
- **ratatui 替代 Rich**: Rust 原生 TUI 渲染, 无外部依赖
- **Tauri 原生窗口替代 OpenTUI**: 不再需要 JS sidecar

---

### 2.10 Skills 技能系统 (28,150 行, 40+ 文件)

> 技能是 **Markdown + YAML 清单** (SKILL.md 文件), **不是 Python 模块**. 但技能系统包含一个复杂的元技能编排器 (DAG 调度器 + 6 个执行器 + Jinja 模板).

#### 架构

**六层覆盖体系** (从低到高优先级):
1. `EXTRA` — 配置额外目录
2. `BUNDLED` — 内置 (`src/opensquilla/skills/bundled/`, 50+ 技能)
3. `MANAGED` — 社区安装 (`$STATE_DIR/skills/`)
4. `PERSONAL` — 用户安装 (`~/.agents/skills/`)
5. `PROJECT` — 工作区 (`{workspace}/.agents/skills/`)
6. `WORKSPACE` — 工作区 (`{workspace}/skills/`)

#### 核心组件

| 文件/目录 | 行数 | 功能 | 迁移策略 |
|-----------|------|------|---------|
| `loader.py` | ~1,000 | 技能加载器: 解析 SKILL.md frontmatter, 构建目录, 文件系统缓存, 热重载 | ✅ `crates/skills/src/loader.rs` (1,438 行) |
| `injector.py` | ~500 | 提示注入: 渲染 `<available_skills>` XML 块到系统提示, Token 预算控制 | ✅ `crates/skills/src/injector.rs` (783 行) |
| `types.py` | ~200 | 核心数据类型: SkillSpec, SkillLayer, SkillRequires | ✅ `crates/skills/src/types.rs` (1,616 行) |
| `eligibility.py` | ~300 | 运行时资格: OS 匹配, 二进制检查 (`shutil.which`), 环境变量 | ✅ `crates/skills/src/eligibility.rs` (923 行) |
| `meta/` | ~5,000 | 元技能编排: DAG 调度器, 6 个执行器, Jinja 模板, 事件流 | ✅ `crates/skills/src/meta.rs` (2,182 行) |
| `hub/` | ~3,000 | 技能分发: ClawHub/GitHub 源, 安装器, 安全扫描器, 锁文件 | ✅ `crates/skills/src/hub.rs` (2,664 行) |
| `creator/` | ~1,000 | 元技能创建工具 | ✅ `crates/skills/src/creator.rs` |
| `bundled/` | 50+ SKILL.md | 内置技能清单 | ✅ 数据可移植 (YAML/Markdown, 无需修改) |

#### 元技能 (Meta-skills)

- `kind: meta` 或 `kind: meta_sop` 定义 DAG 工作流
- `MetaOrchestrator` 拓扑排序步骤, 通过 `asyncio` 并发运行
- 6 种步骤类型: `agent`, `llm_classify`, `llm_chat`, `tool_call`, `skill_exec`, `user_input`
- 支持条件 `when` 表达式, 路由, 双语提示

#### 调度器 (scheduler/, ~6,237 行)

| 文件 | 功能 | 关键点 |
|------|------|--------|
| `engine.py` | `SchedulerEngine` 外观 | ✅ `crates/scheduler/src/engine.rs` |
| `types.py` | 领域类型: CronJob, JobExecution, ScheduleKind (CRON/AT/EVERY) | ✅ `crates/scheduler/src/types.rs` |
| `persistence.py` | `JobStore` — SQLite 持久化 | ✅ `crates/scheduler/src/persistence.rs` |
| `parser.py` | 5 字段 POSIX cron 解析器 | ✅ `crates/scheduler/src/parser.rs` |
| `timer.py` | 精确 `sleep-to-next-due` 滴答循环 | ✅ `crates/scheduler/src/timer.rs` |
| `ops.py` | CRUD 操作, 计划验证, 抖动 | ✅ `crates/scheduler/src/ops.rs` |
| `jobs.py` | 作业执行生命周期, 退避策略, 自动禁用 | ✅ `crates/scheduler/src/jobs.rs` |
| `handlers.py` | 注册的处理器: heartbeat, cron_job, auto_propose, dream | ✅ `crates/scheduler/src/handlers.rs` |
| `delivery.py` | 结果投递链: 信道/WebSocket/Webhook | ✅ `crates/scheduler/src/delivery.rs` |
| `reaper.py` | 过期会话回收 | ✅ `crates/scheduler/src/reaper.rs` |
| `heartbeat.py` | 心跳运行器, SQLite 支持 | ✅ `crates/scheduler/src/heartbeat.rs` |

**调度器关键发现**: 使用纯 Python 自建调度器 (无 APScheduler/croniter). 定时器是纯 `asyncio`, 持久化是 `aiosqlite`. Rust 迁移时可以直接用 `cron` crate + `tokio::time`, 比 Python 版本更简洁.

---

### 2.12 其他模块

| 模块 | 行数 | 功能 | 迁移策略 |
|------|------|------|---------|
| `search/` | 2,722 | 搜索引擎 (Brave, DuckDuckGo, Tavily, Exa, Bocha, IQS) | ✅ `crates/search/src/` (9 文件, 1,417 行) |
| `safety/` | 995 | 注入防护: `<untrusted>` 信封, 工具调用拒绝, 4 类注入模式; 权限矩阵: 3 级风险分类 (SAFE/CONFIRM/ADMIN_ONLY); 密钥脱敏 | ✅ `crates/safety/src/` (4 文件, 1,771 行) |
| `persistence/` | ~500 | 数据库迁移: SQLite 模式迁移, 进程 PID 锁, 预迁移快照备份 | ✅ `crates/persistence/src/` (5 文件, 431 行) |
| `migration/` | ~500 | 历史配置迁移: Hermes, OpenClaw, legacy_detect, env_file | ✅ 一次性迁移脚本, 迁移后删除 |
| `recovery/` | ~1,500 | 崩溃恢复: 事务性恢复, 配置修复, 会话合并, 原子操作, 清理 | ✅ `crates/recovery/src/` (5 文件, 2,190 行) |
| `health/` | ~300 | 健康检查: `build_report()` 统一恢复诊断 | ✅ `crates/health/src/` (5 文件, 352 行) |
| `observability/` | ~2,000 | 可观测性: 结构化日志, 决策日志, 追踪, 提示报告, 安全日志, 用量遥测, 网络策略 | ✅ `crates/observability/src/` (7 文件, 1,739 行) |
| `onboarding/` | ~3,000 | 引导向导: 提供商规格, 信道认证, 配置存储, 探测, 状态机 | ✅ `crates/onboarding/src/` (5 文件, 828 行) |
| `identity/` | ~500 | 终端身份: 启动引导, 提示模板, 工作区, 解析器 | ✅ `crates/identity/src/` (5 文件, 310 行) |
| `eval/` | ~500 | 评测: 集成基准, 合成场景 | ✅ `crates/eval/src/` (5 文件, 1,581 行) |
| `plugins/` | ~100 | TokenJuice 插件 | ✅ `crates/plugins/src/` (3 文件, 154 行) |
| `agents/` | ~300 | Agent 限制/注册表/作用域 | ✅ `crates/agents/src/` (4 文件, 293 行) |
| `compat/` | ~50 | aiosqlite 兼容性 | ✅ 不再需要 |
| `contrib/` | ~500 | CodeTask, SWE-bench 集成 | ✅ `crates/contrib/src/` (4 文件, 247 行) |
| `dist/` | ~100 | 发行版工作区状态 | ✅ `crates/dist/src/` (3 文件, 156 行) |
| `uninstall/` | ~300 | 卸载: 清查, 计划, 安全删除 | ✅ `crates/uninstall/src/` (4 文件, 258 行) |
| `chat/` | ~300 | 对话抽象: conversation, history, source | ✅ `crates/chat/src/` (4 文件, 221 行) |

---

## 三、全 Rust 迁移分层策略

### 架构

```
┌─────────────────────────────────────────────────────────────────┐
│  Tauri 桌面壳 (Rust) — 替换 Electron 12,700 行 main.ts          │
│  ├─ 窗口管理 / 系统托盘 / 深层链接 / 安全存储 / 自动更新        │
│  └─ Tauri shell 命令 (tauri::command) 暴露给 WebUI              │
├─────────────────────────────────────────────────────────────────┤
│  Rust Agent 运行时 (核心二进制, 替代 59,000 行 Python)           │
│  ├─ 层 1: 基础设施 (先迁移)                                     │
│  │  ├─ HTTP/WS 网关 (axum, 替换 app.py/websocket.py)           │
│  │  ├─ 配置管理 (serde + toml, 替换 config.py)                 │
│  │  ├─ SQLite 持久化 (rusqlite, 替换 storage/manager)          │
│  │  ├─ 中间件 (tower, 替换 middleware.py)                       │
│  │  └─ 搜索提供商 (reqwest, 替换 search/)                      │
│  ├─ 层 2: 核心引擎 (并行迁移)                                   │
│  │  ├─ Agent 运行时 (tokio async, 替换 runtime.py)             │
│  │  ├─ Agent 状态机 (Generator trait, 替换 agent.py)           │
│  │  ├─ Turn Runner 8 个 stage (Stage trait, 替换 turn_runner/)  │
│  │  ├─ 工具系统 (22 个内置工具, tokio::process, 替换 tools/)    │
│  │  ├─ 会话管理 (rusqlite, 替换 session/)                      │
│  │  └─ 记忆系统 (rusqlite + FTS5 + candle, 替换 memory/)       │
│  ├─ 层 3: LLM 提供商 (HTTP 协议重写)                            │
│  │  ├─ 5 个后端类 (reqwest + SSE, 替换 provider/)              │
│  │  ├─ Stream assembly (tokio Stream, 替换 stream_assembly.py)  │
│  │  ├─ Text tool normalizer (Rust string, 替换 95K 行)          │
│  │  ├─ 模型路由 (ort + candle, 替换 squilla_router/)            │
│  │  └─ 50+ 提供商注册表 (数据驱动, 替换 registry.py)           │
│  ├─ 层 4: 服务适配器 (后期迁移)                                 │
│  │  ├─ 11 信道适配器 (reqwest + tokio-tungstenite)              │
│  │  ├─ 技能系统 (serde_yaml + tera, 替换 skills/)              │
│  │  ├─ 调度器 (cron crate + tokio, 替换 scheduler/)            │
│  │  ├─ 沙箱 (nix + windows-rs, 替换 sandbox/)                  │
│  │  └─ CLI/TUI (clap + ratatui, 替换 cli/)                     │
└─────────────────────────────────────────────────────────────────┘
```

### 通信方式

```
┌──────────────┐     Tauri invoke() 直接调用     ┌──────────────────┐
│  Vue 3 WebUI │ ◄──────────────────────────────► │  Rust Agent 运行时 │
│  (复用 632 文件)│    tauri::command 命令          │  (同一进程)        │
│               │    事件订阅 (Tauri event)        │                   │
└──────────────┘                                  └──────────────────┘
      │
      │  window.__TAURI__.*
      ▼
┌──────────────────────────────────────────────────────────────────┐
│  Tauri 桌面壳 (窗口, 托盘, 深层链接, 安全存储, 自动更新)          │
└──────────────────────────────────────────────────────────────────┘
```

**关键变化**: 不再有 Python sidecar. 所有模块在同一 Rust 进程内, 通过 Tauri invoke/event 直接与 Vue 3 WebUI 通信. 启动时从单二进制文件加载, 无外部运行时依赖.

### 迁移优先级

| 阶段 | 内容 | 人月 | 累计人月 | 说明 |
|------|------|------|---------|------|
| **P0** | 项目骨架 + 基础设施 | 2-3 | 2-3 | Tauri 项目初始化, axum 网关, serde 配置, rusqlite 迁移, 基本窗口 |
| **P1** | 桌面原生功能 | 1-2 | 3-5 | 窗口/托盘/深层链接/安全存储/自动更新, 前端平台适配层 |
| **P2** | 核心引擎 (最大) | 6-10 | 9-15 | Agent 运行时, Turn Runner, 工具系统, 会话/记忆, MCP |
| **P3** | LLM 提供商 + 模型路由 | 4-6 | 13-21 | 5 个后端类, Stream assembly, Text tool normalizer, 50+ 提供商 |
| **P4** | 服务适配器 | 4-6 | 17-27 | 11 信道, 技能系统, 调度器, 沙箱, CLI/TUI, 可观测性 |
| **P5** | 构建打包 + 签名 | 1 | 18-28 | Tauri bundler, 代码签名, 自动更新, 安装包 |

### 全 Rust 迁移面临的关键挑战

| 挑战 | 难度 | 说明 |
|------|------|------|
| `text_tool_normalizer.py` (95,170 行) | ⭐⭐⭐⭐⭐ | 最复杂的单文件, 需要在 Rust 中复现 DSML/XML/JSON 方言理解和转换逻辑 — ✅ 已完成 `crates/provider/src/text_tool_normalizer.rs` (743 行) + `normalizer.rs` (321 行) |
| `ensemble.py` (171,640 行) | ⭐⭐⭐⭐⭐ | 多模型集成编排器, 纯逻辑但体量巨大, 需完整重写为 Rust trait — ✅ 已完成 `crates/provider/src/ensemble.rs` (3,463 行) |
| `agent.py` (19,290 行) | ⭐⭐⭐⭐ | 显式状态机 ~3000 行, 13 处 subprocess, 大量 asyncio 协同 — ✅ 已完成 `crates/engine/src/agent.rs` (2,449 行) |
| `stream_assembly.py` (30,937 行) | ⭐⭐⭐⭐ | SSE 流解析 + 推理/工具调用缓冲, 需 tokio Stream 实现 — ✅ 已完成 `crates/provider/src/stream_assembly.rs` (726 行) |
| `request_proof.py` (89,921 行) | ⭐⭐⭐⭐ | 预检载荷投影/预算, 大量纯数据转换, 体量大但逻辑简单 — ✅ 已完成 `crates/provider/src/request_proof.rs` (686 行) |
| 沙箱 Linux/macOS/Windows (29,178 行) | ⭐⭐⭐⭐ | 三平台原生 API 调用, 需 nix + windows-rs + seccomp — ✅ 已完成 `crates/sandbox/src/` (5,173 行) |
| 信道 SDK 依赖替代 (Feishu/DingTalk/QQ/Matrix) | ⭐⭐⭐ | 各平台原始协议需要逆向工程, 无现成 Rust SDK — ✅ 已完成 (4,个适配器 700-871 行) |
| 元技能 DAG 编排器 (5,000 行) | ⭐⭐⭐ | DAG 拓扑排序 + 6 个执行器, 算法清晰但体量大 — ✅ 已完成 `crates/skills/src/meta.rs` (2,182 行) |

### Rust crate 依赖清单

| 类别 | crate | 用途 |
|------|-------|------|
| 异步运行时 | `tokio` | 替代 asyncio |
| HTTP 服务 | `axum`, `tower` | 替代 Starlette, 中间件 |
| HTTP 客户端 | `reqwest` | 替代 httpx |
| WebSocket | `tokio-tungstenite` | 替代 websockets |
| 序列化 | `serde`, `serde_json`, `serde_yaml`, `toml` | 替代 Pydantic |
| 数据库 | `rusqlite` | 替代 aiosqlite |
| 模板引擎 | `tera` | 替代 Jinja2 |
| 日志 | `tracing`, `tracing-subscriber` | 替代 structlog |
| CLI | `clap` | 替代 typer |
| TUI | `ratatui` | 替代 Rich |
| ML 推理 | `ort`, `candle`, `ndarray` | 替代 onnxruntime, numpy |
| 分词 | `tokenizers` (HF Rust) | 替代 tiktoken |
| 中文分词 | `jieba-rs` | 替代 jieba |
| 文件监控 | `notify` | 替代 watchdog |
| DNS 解析 | `trust-dns-resolver` | 替代 socket |
| 系统 API | `nix` (Linux), `windows` (Windows) | 替代 ctypes, subprocess |
| 沙箱 | `seccompiler` (Linux) | 替代 seccomp-BPF ctypes |
| 图像处理 | `image` | 替代 Pillow |
| PDF 处理 | `lopdf`, `pdf-extract` | 替代 pdfplumber |
| HTML 解析 | `scraper`, `reqwest` | 替代 readability-lxml |
| XLSX 处理 | `calamine` | 替代 openpyxl |
| 正则 | `regex` | 替代 re |
| Cron | `cron` | 替代自建解析器 |
| 向量搜索 | `sqlite-vec` (C 扩展) | 相同 |
| JSON Schema | `jsonschema` | 替代 Python 验证 |

---

## 四、关键结论

### 4.1 全 Rust 重写规模估算

| 子系统 | Python 行数 | Rust 估算行数 | Rust 当前行数 | 完成度 | 复杂度 | 迁移难度 |
|--------|-----------|-------------|-------------|--------|--------|---------|
| Gateway 网关 | ~6,000 | ~4,000 | 22,641 | 100%+ | 中 | ⭐⭐ |
| Provider 提供商 | ~34,000 | ~25,000 | 25,790 | 100%+ | 高 | ⭐⭐⭐⭐ |
| Engine 引擎 | ~59,000 | ~35,000 | 19,044 | 54% | 极高 | ⭐⭐⭐⭐⭐ |
| Tools 工具 | ~33,000 | ~20,000 | 9,550 | 48% | 高 | ⭐⭐⭐⭐ |
| Session 会话 | ~19,000 | ~10,000 | 7,688 | 77% | 中 | ⭐⭐⭐ |
| Memory 记忆 | ~16,000 | ~10,000 | 8,296 | 83% | 中 | ⭐⭐⭐ |
| Channels 信道 | ~16,000 | ~12,000 | 12,383 | 100%+ | 中高 | ⭐⭐⭐ |
| Sandbox 沙箱 | ~29,000 | ~18,000 | 5,173 | 29% | 高 | ⭐⭐⭐⭐ |
| CLI/TUI | ~36,000 | ~20,000 | 3,881 | 19% | 中 | ⭐⭐ |
| Skills 技能 | ~28,000 | ~15,000 | 10,272 | 68% | 高 | ⭐⭐⭐ |
| Scheduler 调度器 | ~6,000 | ~3,000 | 2,623 | 87% | 低 | ⭐⭐ |
| 其他模块 | ~10,000 | ~6,000 | 15,453 | 93% | 中 | ⭐⭐ |
| Tauri 壳 | — | ~10,000 | 9,952 | 99%+ | 中 | ⭐⭐ |
| **总计** | **~292,000** | **~178,000** | **152,696** | **85.8%** | — | — |

> **注意**: 行数估算基于表 2.1-2.12 中明确的子系统行数累加, 不计交叉引用. Rust 行数估算约为 Python 的 55-65%, 因为 Rust 更简洁的类型系统和零成本抽象可以减少样板代码.

### 4.2 Python 库 → Rust crate 映射

| Python 库 | 行数占比 | Rust 替代 | 迁移难度 |
|-----------|---------|-----------|---------|
| `httpx` (HTTP 客户端) | ~15% | `reqwest` | ⭐ 几乎 1:1 |
| `Starlette` (HTTP 服务) | ~5% | `axum` | ⭐⭐ 概念对应 |
| `Pydantic` (配置验证) | ~10% | `serde` + `validator` | ⭐⭐ 需手动验证规则 |
| `aiosqlite` (异步 SQLite) | ~8% | `rusqlite` + `tokio::task::spawn_blocking` | ⭐⭐ API 不同 |
| `asyncio` (异步) | ~20% | `tokio` | ⭐⭐ 概念对应 |
| `Jinja2` (模板) | <1% | `tera` | ⭐⭐ 语法类似 |
| `structlog` (日志) | ~2% | `tracing` | ⭐⭐ 概念对应 |
| `typer` (CLI) | ~2% | `clap` | ⭐⭐ 声明式 |
| `Rich` (TUI) | ~2% | `ratatui` | ⭐⭐⭐ 范式不同 |
| `Pillow` (图片) | ~1% | `image` | ⭐⭐ API 不同 |
| `pdfplumber` (PDF) | ~1% | `lopdf` / `pdf-extract` | ⭐⭐⭐ 功能不全 |
| `onnxruntime` (ML) | ~3% | `ort` crate | ⭐⭐⭐ API 不同 |
| `numpy` (向量) | ~2% | `ndarray` / `candle` | ⭐⭐⭐ 需重写运算 |
| `jieba` (分词) | <1% | `jieba-rs` | ⭐⭐ 1:1 移植 |
| `tiktoken` (Token) | <1% | `tiktoken-rs` | ⭐⭐ 1:1 移植 |
| `readability-lxml` | <1% | `scraper` + 自实现 | ⭐⭐⭐ 需算法重写 |
| `ctypes` (系统调用) | ~3% | `nix` / `windows-rs` | ⭐⭐⭐⭐ FFI 不同 |
| `yoyo-migrations` (DB 迁移) | <1% | `rusqlite` 自建迁移 | ⭐⭐ 可简化 |

### 4.3 核心架构决策

**全 Rust 重写, 零 Python 运行时依赖.**

```
┌─────────────────────────────────────────────────────────────────┐
│  单二进制文件 (Tauri + Rust Agent 运行时)                        │
│  ├─ Agent 引擎 + Provider + 工具 + 会话/记忆 (core crate)       │
│  ├─ 信道 + 技能 + 调度器 + 沙箱 (plugin crates)                 │
│  ├─ CLI/TUI (clap + ratatui, 可选二进制入口)                    │
│  └─ WebUI 资产 (Vue 3, 编译后嵌入 Tauri 二进制)                 │
└─────────────────────────────────────────────────────────────────┘
```

**技术栈**:
- **语言**: Rust 2024 edition, 稳定版工具链
- **异步**: tokio (单运行时, 所有子系统共享)
- **HTTP 服务**: axum (网关 + 信道 Webhook + 入站 MCP)
- **HTTP 客户端**: reqwest (提供商 + 搜索 + 信道出站)
- **数据库**: rusqlite (所有 SQLite 操作, 含 FTS5 扩展)
- **ML 推理**: ort + candle (ONNX 模型 + 嵌入)
- **序列化**: serde (所有配置 + 协议 + 数据模型)
- **日志/追踪**: tracing + tracing-subscriber + opentelemetry
- **桌面壳**: Tauri v2 (窗口/托盘/深层链接/安全存储/自动更新)

**构建产物**:
- `opensquilla.exe` — 完整桌面应用 (Tauri + WebUI + 核心运行时)
- `osq.exe` — 纯 CLI 模式 (clap, 无 WebUI, 无桌面壳)
- `osq-tui.exe` — TUI 模式 (ratatui, 无 WebUI)

### 4.4 关键数字

| 指标 | 值 |
|------|----|
| Python 总文件数 | 960 |
| Python 总行数 | ~292,000 |
| Rust 估算行数 | ~178,000 |
| 代码量缩减比例 | ~39% |
| 工具数量 | 22 内置 + N 个 MCP |
| LLM 提供商 | 50+ 注册, 5 个后端类 |
| 数据库表 | 19 (会话) + 6 (记忆) |
| 搜索提供商 | 7 (Brave, DuckDuckGo, Tavily, Exa, Bocha, IQS) |
| 信道 | 11 (Slack, Discord, Telegram, Feishu, DingTalk, QQ, WeCom, Matrix, Teams, Terminal, WS) |
| RPC 处理器 | 30+ |
| 配置模型 | 30+ serde 子模型 |
| 引擎核心 | 59,126 行 (最大模块) |
| 沙箱行数 | 29,178 行 (最复杂非功能性模块) |
| Web UI 源文件 | 632 (Vue 3, **100% 可复用**) |
| 预估迁移总人月 | **18-28 人月** (全栈 Rust 团队, 含测试 + 调试) |
| 预估团队规模 | 3-5 人全职 |
| 预估迁移日历 | **6-12 个月** |
| 运行时依赖 | 无 (单二进制, 原生编译) |
| 二进制大小 | ~50-80 MB (含 WebUI 资产) |
| 内存占用 | ~30-50 MB (不含 LLM 推理) |