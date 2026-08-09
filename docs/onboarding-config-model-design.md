# onboarding.* Rust 移植 — 配置模型扩展与引擎改造 详细设计文档

> 状态:设计中(2026-08-09)
> 分支:rust-migration-skeleton
> 决策:方案 A —— **扩展 `opensquilla_core::config::Config` 结构体(新增 Python 形状字段),并改造引擎各消费点**。
> 配套:调研文档 [onboarding-rust-parity-gap.md](./onboarding-rust-parity-gap.md)(问题面/缺口矩阵);本文档只写**实施设计**,供实现与后续查证。

---

## 0. 文档目的与阅读对象

- 本仓库是 Python 网关 → 全 Rust 的迁移。前端的设置向导(opensquilla-webui/src/composables/setup/**)按 **Python 配置形状**(`llm.*`、`llm_profiles.*`、`search.*`、`image_generation.*`、`memory.*`、`audio.*`、`squilla_router.*`、`llm_ensemble.*`)读写配置。
- 已确证:前端 `config.llm` 在 Tauri 桌面端恒为 `undefined`(`get_config` 直接序列化 Rust `Config`,而 Rust `Config` 无 `llm` 字段);`Config::set("llm.provider", …)` 经 `serde_json::from_value::<Config>` 回灌时**未知键被丢弃**;`patch_config` 只写 `ConfigStore` 内存 overlay、不落盘、`get_config` 也读不到。
- 结论:仅"移植 onboarding RPC 方法"不够,必须先把 **Rust 配置模型扩展到能表达 Python 形状**,再让引擎消费新模型,最后 onboarding 写路径落盘可回读。
- 本文档给出:配置模型字段规格(§2)、引擎改造面(§3)、持久化/加载设计(§4)、onboarding 移植计划(§5)、副作用同步矩阵(§6)、前端桥接(§7)、迁移与兼容(§8)、分阶段落地与验证(§9)、风险登记(§10)。

---

## 1. 核心决策与总览

### 1.1 为什么必须扩展 typed Config(而非 overlay 视图层)

曾评估两条路线并最终按用户决策选 A:

| 路线 | 做法 | 结论 |
|---|---|---|
| B. overlay 视图层 | onboarding 写 `ConfigStore.entries` overlay + 镜像 `providers[]`;`get_config` 返回 merged | 被否:双写冗余、overlay 永不落盘需另建持久化、引擎与前端两套真相,长期不可维护 |
| **A. 扩展 typed Config** | `Config` 新增 `llm`/`llm_profiles`/`search`/`image_generation`/`memory`/`audio`/`squilla_router`/`llm_ensemble` 字段,引擎消费新字段,onboarding 直写 `Config` 并落盘 | **选定**;单一事实来源,与 Python `GatewayConfig` 对齐,前端形状天然满足 |

### 1.2 关键不变量(实现必须遵守)

1. **单一事实来源**:`Config` 是唯一持久化配置模型;废弃 `ConfigStore.entries` overlay 作为业务数据载体(或仅保留为临时 override,不再承担 onboarding 数据)。
2. **前端形状不变**:`get_config` 返回的 JSON 顶层键必须包含前端要求的 `llm`/`llm_profiles`/`search`/`image_generation`/`memory`/`audio`/`squilla_router`/`llm_ensemble`/`channels`/`providers`,字段名与 Python `GatewayConfig` 一致(camelCase 由序列化层处理,见 §4)。
3. **写必落盘**:所有 onboarding 写方法成功后 `Config::save()`;重启不丢。
4. **密钥不落盘明文**:`api_key` 若来自 env,序列化跳过;写盘走红action(§4.4)。
5. **老文件兼容**:存量 `opensquilla.toml`(现为 `providers[]` 结构)解析不能失败;新字段 `#[serde(default)]` 全量可选(§8)。
6. **写顺序对齐 Python**:`persist 优先 → apply_inplace → sync 副作用`(调研文档 §2.6)。

### 1.3 总体改动文件面(已调研收敛)

- `crates/core/src/config.rs`:新增 8+ 配置段结构体与 `Config` 顶层字段(§2)、`save_to` 原子写、密钥红action(§4)。
- `src-tauri/src/agent_bridge.rs`:`resolve_provider`/`list_providers`/状态 优先读 `llm`(§3.3);`get_config`/`patch_config`/`set_config` 走统一 Config(§4.2);新增 onboarding 命令。
- `crates/gateway/*`:onboarding RPC 处理器(重写 onboarding.rs)、`validate`/`router_config_from_config` 扩展、`ConfigStore` 持久化对齐。
- `src-tauri/src/main.rs` + `commands.rs` + `state.rs`:启动兜底补新字段、onboarding 命令注册、`config_store` 初始化对齐。
- `opensquilla-webui/src/lib/rpc.ts`:`TAURI_METHOD_REGISTRY` 补 onboarding 绑定。
- `crates/cli/*`:default_provider/default_model 读 `llm`;providers add/remove/set_default 同步 llm(§3.3)。
- `crates/engine/*`:**基本无需改动**(引擎不读 Config,见 §3.1)。

---

## 2. 配置模型字段规格(Python → Rust 映射)

> 依据 Python `src/opensquilla/gateway/config.py` 的 pydantic 模型(3352 行)。关键结论:**`GatewayConfig` 无顶层 `providers` 数组,`llm` 即主 provider**;`image_generation.providers` / `audio.providers` 是内嵌字典。

### 2.1 映射规则

- 段名对齐 Python 顶层键:`llm`、`llm_profiles`、`search_*`(内联)、`image_generation`、`audio`、`memory`、`squilla_router`、`llm_ensemble`、`channels`。
- Rust 字段 snake_case,命令返回层 `#[serde(rename_all = "camelCase")]` 对齐前端(§4.3)。
- 全量 `#[serde(default)]` + `skip_serializing_if = "Option::is_none"`,保证老文件兼容与最小写盘。
- 枚举用 Rust 枚举 + `#[serde(rename_all = "...")]`,未知值容错反序列化(前端可能发旧值)。
- **scope**:本设计只覆盖 onboarding 相关段(前端 setup 读写的键)。其余 GatewayConfig 字段(tls/auth/cors/rate_limit/task_runtime/model_catalog/prompt/safety/mcp/heartbeat/agents/sandbox/… 前端不读)按需后续补充;`get_config` 返回缺这些键对前端无影响(前端类型全可选)。

### 2.2 Rust 新增字段规格(承重内容)

`Config` 顶层新增(全部 `#[serde(default, skip_serializing_if = "Option::is_none")]`):

```rust
pub llm: Option<LlmConfig>,                       // [llm] 主 provider
pub llm_profiles: Option<HashMap<String, LlmProfile>>, // [llm_profiles.<id>]
pub llm_ensemble: Option<LlmEnsembleConfig>,      // [llm_ensemble]
pub search_provider: Option<String>,              // search 内联字段(与 Python 对齐)
pub search_api_key: Option<String>,
pub search_api_key_env: Option<String>,
pub search_max_results: Option<u32>,
pub search_proxy: Option<String>,
pub search_use_env_proxy: Option<bool>,
pub search_fallback_policy: Option<String>,       // "off"|"network"
pub search_diagnostics: Option<bool>,
pub image_generation: Option<ImageGenerationConfig>, // [image_generation]
pub audio: Option<AudioConfig>,                   // [audio]
pub memory: Option<MemoryConfig>,                 // [memory]
pub squilla_router: Option<SquillaRouterConfig>,  // [squilla_router]
```

各段结构体字段(Python config.py 行号):

**LlmConfig**(Python `LlmProviderConfig`, config.py:360)

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| provider | String | "tokenrhythm" | 主 provider id |
| model | String | "deepseek-v4-pro" | deepseek 前缀归一化 |
| api_key | Option<String> | None | 明文,落盘红action(§4.4) |
| api_key_env | Option<String> | None | env 变量名(非敏感) |
| base_url | String | "https://tokenrhythm.studio/v1" | |
| proxy | Option<String> | None | HTTP 代理 |
| max_tokens | u32 | 0 | 0=目录自动 |
| context_window_tokens | u32 | 0 | |
| temperature / top_p | Option<f64> | None | |
| thinking | Option<String> | None | 兼容 thinking/thinking_level |
| provider_routing | HashMap<String,String> | {} | OpenRouter model→upstream |

**LlmProfile**(Python `LlmProviderProfile`, config.py:2025, `extra="ignore"`)

| 字段 | 类型 | 说明 |
|---|---|---|
| model / api_key / api_key_env / base_url / proxy | Option<String> | |
| api_key_env_pool | Vec<String> | env 变量名池 |

**LlmEnsembleConfig**(config.py:502):enabled(bool,false)/mode("b5_fusion")/selection_mode(String)/proposer_tools(bool,false)/min_successful_proposers(u32,1)/all_failed_policy("fallback_single"|"error")/model_options(Vec<String>)/candidates(Vec<EnsembleCandidate>)/candidate_max_chars(u32,24000)/proposer_timeout_seconds & aggregator_timeout_seconds(f64,3600)/shuffle_candidates(bool,true)/record_candidates(bool,false);`EnsembleCandidate`(config.py:458):provider/model(必)/source("custom"|"legacy_model_options")/enabled(true)/role("")/thinking_level("")。

**ImageGenerationConfig**(config.py:1470):enabled(bool,false)/binding("custom"|"follow_llm")/primary("openai/gpt-image-1")/fallbacks(Vec<String>)/size("1024x1024")/timeout_seconds(u32,180)/output_format(Option<String>)/providers(HashMap<String, ImageGenProvider>);`ImageGenProvider`:base_url/api_key/api_key_env(均 Option)。

**AudioConfig**(config.py:1520):enabled(bool,false)/tts(AudioTtsConfig:model/voice/language_code/output_format/timeout_seconds/stability/similarity_boost/style/use_speaker_boost/speed)/providers(HashMap<String, AudioProvider>);`AudioProvider`(仅 elevenlabs):base_url/api_key/api_key_env/speech_to_text_model/voice_conversion_model/music_model/music_output_format。

**MemoryConfig**(config.py:911,超大,onboarding 只读写 `memory.embedding`):embedding(MemoryEmbeddingConfig)/cost/source/retrieval_mode/…;`MemoryEmbeddingConfig`(config.py:877):provider("auto")/mode/model/api_key/base_url(均 Option)/local{onnx_dir}/remote{api_key,api_key_env,base_url,model,headers,dimensions}/ollama{base_url,model}。

**SquillaRouterConfig**(config.py:1135,超大):enabled(true)/auto_thinking(true)/tier_profile/preset_binding/visual_mode/cross_provider_tiers/tier_provider_mismatch/tiers(HashMap)/default_tier/strategy/rollout_phase/confidence_threshold(0.5)/…(onboarding 只读写 router.configure 的参数子集,其余保留默认)。

**channels**:现有 `Config.channels: Vec<ChannelConfig>` 保留;Python 是标签联合 `ChannelConfigEntry`(type 判别),前端 channels 操作读写 `channels` 段——字段级对齐见调研文档 §2.2 channels 条目,实现时按 channel type 子结构建 enum。

**llm 与 providers 关系**:`llm` = 主 provider(真相);现有 `providers: Vec<ProviderConfig>` 保留为引擎运行列表,由 §3 D3 定义单向派生。

### 2.3 序列化细节(对齐 Python)

- Python `model_dump(exclude_none=True, exclude_defaults=False)` + `redact_public_config`(config.py:3302)屏蔽敏感字段值 → Rust 落盘用同规则:`api_key` 类字段按 §4.4 处理;`*_env` 字段**不敏感**,保留可读。
- Python 空 secret 剔除 dump → Rust `skip_serializing_if = "Option::is_none"` 等价。

---

## 3. 引擎改造面

> 依据已确认调研事实。**核心发现:引擎(`crates/engine`)根本不读 `Config`** —— 主 provider 解析全在 src-tauri 的 `resolve_provider`,且每次消息重读,无缓存。这让"改造引擎"的实际面大幅收窄。

### 3.1 已确认事实

- **引擎不读 Config**:`crates/engine` 内无 `Config.providers` 读取点;引擎用自有 `TurnRunnerConfig { default_provider, default_model }`(`crates/engine/src/turn_runner/mod.rs:115/140`);`AgentRuntime::new` 不接收 config(`main.rs:332`)。
- **主 provider 实时解析在 src-tauri**:`resolve_provider`(`agent_bridge.rs:1580-1642`):`name=="default"` 取 `providers.first()`(:1586),否则 `find_provider(name)`(:1589),按 provider_type 分发 OpenAiCompat/Anthropic/Ollama(:1619-1633),读 api_key/base_url(:1593)。
- **无缓存/无 reload 机制**:`reload_config`(`main.rs:67`)只读回序列化;主 provider 切换靠每次 `send_message` 重读 `state.config()`(`agent_bridge.rs:650`)重建 generator → **写配置后下一消息即生效**。
- **其他读点**:`list_providers` 默认 provider = `providers.first()`(:1023-1024)、provider 状态列表(:1383-1384)、`provider_status_json`(:1316-1324);cli `util.rs:77-102`(default_provider/default_model)、`util.rs:120-129`(build_provider_registry 遍历 providers)、`providers.rs:197/216/233`(add/remove/set_default,靠"移到位置0"表达主 provider);gateway `config.rs:150-188 validate` + `:335-363 router_config_from_config`。
- **`Config::set` 点号调用方**:`crates/recovery/repair.rs`、`crates/onboarding/flow.rs`、`crates/onboarding/storage.rs`、`cli/cost.rs:589`、`cli/router.rs:761`。这些用旧 Python 点号键(`provider.openai.api_key`、`model.default`),需在 §8 迁移中确认映射。

### 3.2 决策点(已按事实敲定)

- **D1 主 provider 真相来源 = 新 `llm` 字段**:`resolve_provider` 优先读 `config.llm`(provider/model/api_key/base_url),缺失时回退 `providers.first()`。影响面:`agent_bridge.rs:1586/1589/1024/1384`、`cli/util.rs:77-102`、`gateway config.rs:335`。**引擎无需改动**(它本来就不读 Config)。
- **D2 热更新 = 每消息生效,无需 reload 机制**:既有 per-message 重建已满足"写配置后生效";若某副作用(Python `broadcast_model_routing_changed`)需要即时推送,再按 §6 单独接线。
- **D3 派生方向**:`llm` = 主 provider 真相;`providers[]` 保留为引擎/CLI 运行列表,由 onboarding 写方法在写 `llm` 时**同步维护**(新增/更新首元素或按名 upsert)。禁止反向(以 providers[0] 为准覆盖 llm),避免双真相。

### 3.3 具体改动清单(初版)

| 文件 | 改动 |
|---|---|
| `agent_bridge.rs:1580-1642` | `resolve_provider` 优先 `llm` → 回退 `providers.first()`;`default_model` 优先 `llm.model` |
| `agent_bridge.rs:1023/1384` | `list_providers`/状态列表的 default_provider 语义随 llm |
| `cli/util.rs:77-102`、`providers.rs` | default_provider/default_model 同 llm 优先;add/remove/set_default 改为写 `llm` + 同步 `providers[]` |
| `gateway/config.rs:150-363` | validate 扩展 llm 必填校验;`router_config_from_config` 读 `llm.model` |
| `crates/onboarding/flow.rs`、`storage.rs` | 旧点号键(`provider.*`/`model.*`)迁移到 `llm`/`llm_profiles`(见 §8) |

---

## 4. 持久化 / 加载 / 序列化设计

> 依据已确认调研事实(crates/core/src/config.rs、src-tauri/src/main.rs、crates/gateway/src/config.rs)。

### 4.1 现状事实(已确证)

- 启动:`main.rs:91` `Config::load()`,失败时 `:98-108` **手写默认 `Config` 字面量**(新增字段时需同步补);`build_app_state`(`main.rs:291-342`)把 config 传入 `AppState::new`(`state.rs:65-81`)→ `config: Arc<RwLock<Config>>`(:73)。
- `config_store: ConfigStore::new()`(`state.rs:77`)**空默认,未用真实 config 初始化**。
- `AgentRuntime` 不含 config;网关只在 `runtime.rs:97-99` `DesktopRuntime::from_config` 取 `.gateway.clone()` 起步——**引擎与网关当前都不持有完整 Config**。
- `save_to`(`config.rs:186-194`)= `toml::to_string` + `fs::write`,**非原子,无临时文件+rename,无权限/回滚**。
- 密钥:**零红action机制**。`ProviderConfig.api_key`(`config.rs:293`)= `Option<String>` + `skip_serializing_if`,Some 则**明文写 TOML**。crates/core 无 `api_key_env`/`redact` 命中。

### 4.2 统一写路径(设计)

- 所有写命令(`set_config`/`patch_config`/onboarding 写方法)收敛到同一个"改 `Config` → `Config::save()`"入口。
- `Config::save_to` 改为**原子写**:临时文件 + `fs::rename`(+ 尽量 0600 权限)。对齐 Python `persist_config` 语义。
- **`patch_config` 落盘**:从"只写 `config_store.patch`(内存)"改为写入统一 Config 并落盘。
- **单一数据源**:废弃 `ConfigStore.entries` 作为业务数据载体;`get_config`/`get_config_effective` 读同一个 Config。`ConfigStore` 若保留,用 `ConfigStore::from_config(real_config)`(`gateway/config.rs:43-50`)初始化,或让其内部直接引用同一 Config。
- **`get_config` 改道**:`agent_bridge.rs:1174` 现读 `state.config`;合并语义与 `get_config_effective`(`:1397`,合并 overlay)统一。

### 4.3 序列化命名

- TOML 文件字段保持 snake_case(对齐 Python 格式 `[llm]` 段)。
- 命令返回 struct 统一 `#[serde(rename_all = "camelCase")]`(与现有 `ProviderInfo`/`ConfigSetResponse` 一致)。
- 参考现有 `ControlUiConfig`(`config.rs:459`)已用 `#[serde(rename_all="camelCase", default)]` 的先例。

### 4.4 密钥红action(新增需求)

- 新增 `api_key_env: Option<String>` 字段:来源为 env 时只记变量名,不落盘明文。
- `api_key` 落盘策略对齐 example 约定(env 变量名注释);env 来源时序列化跳过 `api_key`。
- 需自定义 `Serialize`/`Deserialize` 掩码或字段级处理,避免明文写盘。

### 4.5 格式兼容关键事实

`opensquilla.toml.example` 是**旧 Python 格式**(`[llm]`/`[llm.provider_routing]`/`[llm_ensemble]`/`[models.vllm.*]`/`[memory]`/`[sandbox]`/`[permissions]`),与 Rust `Config` 现有字段(`gateway/providers[]/channels/models/sandbox/skills/scheduler/observability/control_ui`)几乎不重叠。→ **Rust typed Config 从未能解析真实格式文件**。方案 A 扩展 Config 对齐 Python 格式后,example 文件才可被读取;现有 `providers[]` 段与 Python 格式的关系在 §8 迁移中定义。

---

## 5. onboarding 移植计划(写路径)

移植顺序与依赖(对齐调研文档 §6 建议,但落在"扩展 Config"之上):

1. **P0 配置模型扩展 + 持久化统一**(前置,§2+§4)。
2. **P0 只读救场**:`onboarding.catalog`、`onboarding.status`(让设置界面能打开)。
3. **P1 写方法面**(依赖顺序):`provider.configure` → `llmProfile.upsert/activate/remove` → `router.configure` → `ensemble.configure` → `search/imageGeneration/memory_embedding/audio.configure` → `capability.reset` → `channel.upsert/remove/enable/disable`。
4. **P2 probe/discover**:`provider.probe`、`llmProfile.probe/draft.probe`、`models.discover` 系列、`credential.reveal`。
5. **P3 前端注册表**:`rpc.ts` 补绑定;`supportsMethod`/`profileSaveSupported` 自动恢复。

每个写方法的入参/返回/底层 Python 实现位置,已在调研文档 §2.1/§5 表格逐条列出,实现时逐条对照。

---

## 6. 副作用同步矩阵

Python 写方法调用 13 个 gateway 服务(调研文档 §2.6 清单)。Rust 侧需逐一确认对应物:

| Python 副作用 | Rust 对应物 | 状态(待确认) |
|---|---|---|
| `llm_runtime.resolve_llm_credential` | ? | 待盘点 |
| `config_secrets.inherit_runtime_secrets` | ? | 待盘点 |
| `model_routing.broadcast_model_routing_changed` | `ModelRouter`(config.rs:18) | 已存在,接线待确认 |
| `model_catalog_refresh.refresh_live_model_catalog` | `LiveCatalog`(provider/src/live_catalog.rs) | 已存在,接线待确认 |
| `channels_bridge` 热重载 | `channels`(gateway channels handler) | 待确认 |
| `provider.selector.sync_primary` | ? | 待盘点 |
| `tools.builtin.media/web` 热配置 | ? | 待盘点 |
| `engine.selector_override` / `usage_accounting` | ? | 待盘点 |

无对应物的标为**文档化偏差**,在前端侧降级(现有 supportsMethod 守卫机制)。> 注:此表在 S4 实现时逐项盘点并更新(每个写方法对应 Python 侧调用的副作用,在 Rust 网关确认存在性)。

---

## 7. 前端桥接(§7 简要)

- `TAURI_METHOD_REGISTRY`(opensquilla-webui/src/lib/rpc.ts)补 27 条 onboarding 绑定,每条必带 `command`(TS2741)。
- `TAURI_SUPPORTED_METHODS` 自动随 registry 键扩展 → `supportsMethod('onboarding.llmProfile.upsert')` 恢复 true → `profileSaveSupported` 走新分支。
- 事件通道:`agent:stream:*` 已有;onboarding 无独立事件需求(写方法同步返回)。

---

## 8. 迁移与兼容

### 8.1 双格式并存(已确证)

- `opensquilla.toml.example` 是**旧 Python 格式**(`[llm]`/`[llm.provider_routing]`/`[llm_ensemble]`/`[models.vllm.*]`/`[memory]`/`[sandbox]`/`[permissions]`),Rust `Config` 现有结构**无法解析它**。
- 方案 A 扩展 Config 对齐 Python 段后,该文件可被读取(`[llm]` → `llm` 字段)。
- Rust 形状文件(`providers[]`/`channels[]`)仍可解析:新字段全 `Option` + 缺省 → `None`,不影响反序列化。
- 两者并存期:**`llm` 为主 provider 真相**,`providers[]` 为运行列表;加载时若 `llm` 缺失而 `providers[0]` 存在,迁移提升 `providers[0]` → `llm`(反向填充)。

### 8.2 旧点号键迁移

`Config::set` 调用方(`crates/recovery/repair.rs`、`crates/onboarding/flow.rs`、`crates/onboarding/storage.rs`、`cli/cost.rs:589`、`cli/router.rs:761`)用 `provider.openai.api_key`/`model.default` 等旧 Python 点号键。迁移映射:

| 旧点号键 | 新键 |
|---|---|
| `provider.<id>.api_key` | `llm.api_key`(主)/ `llm_profiles.<id>.api_key`(profile) |
| `model.default` | `llm.model` |

`Config::set` 的 `to_value_map`/`apply_value_map` serde 往返对新增顶层字段自动兼容(§3 事实),无需改动其机制;只需调用方改用新键。

### 8.3 兼容约束

- 新增字段全部 `#[serde(default)]`;`ProviderConfig` 已有 5 个必填字段无 serde default → **新字段必须 `Option` + `skip_serializing_if`**,否则破坏旧文件反序列化(见 §2.1 映射规则)。
- `main.rs:98-107` 手写兜底 `Config` 字面量需同步补新字段。
- CLI 既有 `providers[]` 命令(add/remove/set_default)保持可用,语义对齐 llm(§3.3)。

---

## 9. 分阶段落地与验证

| 阶段 | 内容 | 提交粒度 | 验证 |
|---|---|---|---|
| S1 | Config 模型扩展(纯新增字段,不改行为) | 1 commit | 静态 + CI 编译 |
| S2 | 持久化统一(patch/set 落盘 + get_config 读同一 Config) | 1 commit | CI + 手工配置写读 |
| S3 | 只读 onboarding.catalog/status | 1 commit | CI + 前端界面可开 |
| S4 | 写方法面(按 §5 顺序,分块提交) | 每 2-3 方法 1 commit | CI + 前端保存验证 |
| S5 | probe/discover/credential | 分块 | CI |
| S6 | rpc.ts 绑定 + 前端降级分支清理 | 1 commit | vue-tsc + 手工 |

验证约束:**禁止本地 cargo/npm 构建**,静态复核 + `gh` CLI 触发/查看远程 workflow(`Tauri Build & Upload` 三 OS),按 §5 缺口矩阵四维对齐(方法名/参数形状/返回形状/事件通道)人工审计。

---

## 10. 风险登记

| 风险 | 等级 | 缓解 |
|---|---|---|
| 主 provider 真相来源切换(llm vs providers[0]) | 高 | D1/D3 已敲定(llm 为真相,resolve_provider 优先 llm 回退 providers[0]);派生逻辑单测 |
| 新增字段的读点散布 agent_bridge/cli/gateway validate,遗漏改造成 bug | 中 | 消费点清单已列 §3.3;S1 纯新增不改行为 |
| TOML/YAML/JSON 多格式序列化不一致 | 中 | §4 序列化规则统一;S2 验证读写回环 |
| probe 真实网络请求在 CI 不可跑 | 中 | probe 返回结构单测 + fail-closed;CI 只编译 |
| 前端 27 方法绑定遗漏 | 中 | §5 表格逐条核对;vue-tsc 只查前端 |

---

## 附:与调研文档的关系

- 本文件 = **实施设计**(怎么做);`onboarding-rust-parity-gap.md` = **调研事实**(是什么/在哪)。
- 实现时以本文件 §2/§5 为准,事实出处引调研文档 §2/§5。
