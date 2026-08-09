# onboarding.* Rust 移植缺口 — 技术调研文档

> 状态:调研完成(2026-08-09,并行只读子代理测绘 + 主代理静态复核)
> 分支:rust-migration-skeleton
> 结论先行:**onboarding.\* 不是"契约不匹配",而是 Rust 侧整个方法面从未实现/未注册。前端调用的 27 个 onboarding.\* 方法全部 METHOD_NOT_FOUND;Rust 网关里的 onboarding.rs 是一套完全不同的简化 API,且从未被 Gateway::new 挂载(死代码);配置持久化还有独立缺陷(patch_config 只写内存 overlay 不落盘)。**
>
> 实施设计见 [onboarding-config-model-design.md](./onboarding-config-model-design.md)(方案 A:扩展 typed Config + 改造引擎)。本文档只记录"是什么/在哪"的调研事实。

---

## 1. 问题本质(五层拆解)

| 层 | 现象 | 证据 |
|---|---|---|
| 1. 功能整体缺失 | `onboarding.*` 整个方法面在 Rust 侧从未实现 | `src-tauri` 命令面只有只读 provider 命令;gateway RPC 未注册 onboarding;前端 `rpc.call('onboarding.*')` 全部抛 `METHOD_NOT_FOUND`,请求到不了 Rust |
| 2. 双重重断 | 前端注册表无 onboarding.* 绑定 + Rust 无写命令 | `opensquilla-webui/src/lib/rpc.ts:864-1069` 的 `TAURI_METHOD_REGISTRY` 共 52 条,0 条 onboarding;唯一写入口是通用 `patch_config`(且不落盘) |
| 3. Rust 侧只有死代码 | `crates/gateway/src/onboarding.rs` 实现了一版简化 wizard,但从未接入 | `Gateway::new`(`crates/gateway/src/app.rs:93-163`)与 `register_domain_handlers`(`crates/gateway/src/rpc_handlers.rs:97-111`)均未调用 `register_onboarding_handlers`;方法面(providers/start/select_provider/set_api_key/...)与前端要的 camelCase 完全不同 |
| 4. 真实实现只在 Python | camelCase 的 onboarding.* 只在 `src/opensquilla/` | 30 个 RPC 方法分布在 `gateway/rpc_onboarding.py` + `onboarding/{mutations,status,probe}.py` + 7 个 spec 模块,合计约 1 万行 |
| 5. 失败方式有欺骗性 + 持久化独立缺陷 | METHOD_NOT_FOUND 被前端 catch 成 toast;supportsMethod 恒 false 走未桥接旧分支;patch_config 只写内存 overlay | 详见 §4.4 与 §5.3 |

用户侧感知:设置向导打不开(`Failed to load settings` toast)、provider 保存静默失败、Channels 操作全失败、ChatView mic 按钮恒不可用。

---

## 2. Python 参照实现测绘(移植目标)

### 2.1 onboarding.* 方法面 —— 30 个方法(全部在 `src/opensquilla/gateway/rpc_onboarding.py`)

通过 `@_d.method("onboarding.<name>", scope="...")` 注册。12 个命名空间:

**provider 命名空间**
| camelCase 方法名 | rpc_onboarding.py:行 | 输入参数 | 返回关键字段 |
|---|---|---|---|
| `onboarding.status` | :412 | 无 | configPath/hasConfig/llmConfigured/llmSource/llmEnvKey/llmCredentialStatus/llmProfileStatus/searchConfigured/imageGenerationConfigured/audioConfigured/memoryEmbeddingConfigured/channelCount/channelsConfigured/ensembleCredentialStatus/needsOnboarding/sections/sectionDetails/warnings |
| `onboarding.catalog` | :417 | 无 | providers/channels/searchProviders/routerProfiles/memoryEmbeddingProviders/imageGenerationProviders/audioProviders |
| `onboarding.provider.configure` | :579 | providerId(必), model/apiKey/apiKeyEnv/preserveApiKey/baseUrl/proxy/presetId/routerAction/imageGenerationIntent | changed/restartRequired/configPath/entry/warnings |
| `onboarding.provider.probe` | :1187 | providerId(必), model/apiKey/apiKeyEnv/baseUrl/proxy | ok/providerId/model/failureKind/message/code/latencyMs/firstResponseMs/totalMs |
| `onboarding.provider.credential.reveal` | :1253 | providerId(必) | ok/provider/source/envKey/apiKey |
| `onboarding.provider.credential.clear` | :1259 | providerId(必) | changed/restartRequired/configPath/entry/warnings |
| `onboarding.models.discover` | :1307 | providerId(必), apiKey/apiKeyEnv/baseUrl/proxy/forceRefresh | result.to_payload()(模型列表) |

**llmProfile 命名空间**
| camelCase 方法名 | 行 | 输入参数 | 返回 |
|---|---|---|---|
| `onboarding.llmProfile.upsert` | :629 | providerId(必), model/apiKey/apiKeyEnv/apiKeyEnvPool/keepCurrentSecret/baseUrl/proxy | changed/restartRequired/configPath/entry/warnings |
| `onboarding.llmProfile.credential.clear` | :681 | providerId(必) | 同上 |
| `onboarding.llmProfile.remove` | :723 | providerId(必) | 同上 |
| `onboarding.llmProfile.active.remove` | :747 | providerId(必), replacementProviderId(必), replacementModel/routerAction/imageGenerationIntent | 同上 |
| `onboarding.llmProfile.activate` | :833 | providerId(必), model/routerAction/imageGenerationIntent | 同上 |
| `onboarding.llmProfile.probe` | :1043 | providerId(必), model(必) | probe payload(同 provider.probe) |
| `onboarding.llmProfile.draft.probe` | :1080 | providerId(必), model(必), apiKey/apiKeyEnv/baseUrl/proxy/keepCurrentSecret | probe payload |
| `onboarding.llmProfile.models.discover` | :1114 | providerId(必), forceRefresh | 模型列表 |
| `onboarding.llmProfile.draft.models.discover` | :1151 | providerId(必), apiKey/apiKeyEnv/baseUrl/proxy/keepCurrentSecret/forceRefresh | 模型列表 |

**router / ensemble / channel 命名空间**
| camelCase 方法名 | 行 | 输入参数 | 返回 |
|---|---|---|---|
| `onboarding.router.catalog` | :1392 | 无 | router_catalog_payload() |
| `onboarding.router.configure` | :1399 | mode/defaultTier/tiers/crossProviderTiers/tierProviderMismatch | changed/restartRequired/configPath/entry/warnings |
| `onboarding.ensemble.configure` | :1439 | enabled/selectionMode/modelOptions/candidates/minSuccessfulProposers/allFailedPolicy | 同上 |
| `onboarding.channel.probe` | :1478 | entry(对象,必) | status/connected/probeKind/restartRequired/entry/warnings |
| `onboarding.channel.upsert` | :1802 | entry(对象,必) | changed/restartRequired/liveApply/configPath/entry/warnings |
| `onboarding.channel.remove` | :1827 | name(必) | changed/restartRequired/liveApply/configPath/removed |
| `onboarding.channel.enable` | :1869 | name(必) | changed/restartRequired/liveApply/configPath/name/enabled |
| `onboarding.channel.disable` | :1874 | name(必) | 同上 |

**search / imageGeneration / memory_embedding / audio / capability 命名空间**
| camelCase 方法名 | 行 | 输入参数 | 返回 |
|---|---|---|---|
| `onboarding.search.configure` | :1510 | providerId(必), apiKey/apiKeyEnv/maxResults/proxy/useEnvProxy/fallbackPolicy/diagnostics | changed/restartRequired/configPath/entry/warnings |
| `onboarding.imageGeneration.configure` | :1544 | providerId(必), primary/apiKey/apiKeyEnv/baseUrl/enabled/size/outputFormat/fallbacks/clearFallbacks/credentialMode | 同上 |
| `onboarding.imageGeneration.models.discover` | :1371 | providerId(必) | 模型列表 |
| `onboarding.memory_embedding.configure` | :1586 | providerId(必), model/apiKey/apiKeyEnv/baseUrl/onnxDir | 同上 |
| `onboarding.audio.configure` | :1703 | providerId(必), apiKey/apiKeyEnv/baseUrl/enabled/ttsVoice/ttsModel/languageCode | 同上 |
| `onboarding.capability.reset` | :1720 | capabilityId(必) | 同上 |

> 注:用户消息称"18 个写入方法",实际 Python 侧 30 个方法(含 status/catalog/router.catalog 3 个只读),前端实际调用 27 个不同方法名(不含 router.catalog)。文档以实际测绘为准。

### 2.2 catalog 汇聚逻辑(7 个 spec 模块)

`rpc_onboarding.py:417` 的 `_onboarding_catalog` 聚合:

```python
return {
    "providers": provider_catalog_payload(),                        # provider_specs.py:458
    "channels": channel_catalog_payload(),                          # channel_specs.py
    "searchProviders": search_provider_catalog_payload(),           # search_specs.py:155
    "routerProfiles": router_catalog_payload(),                     # router_specs.py:72
    "memoryEmbeddingProviders": memory_embedding_provider_catalog_payload(),  # memory_embedding_specs.py:117
    "imageGenerationProviders": image_generation_provider_catalog_payload(),  # image_generation_specs.py:119
    "audioProviders": audio_provider_catalog_payload(),             # audio_specs.py:156
}
```

各 payload 字段:
- **providers**:providerId/label/backend/providerKind/runtimeSupported/verification/envKey/defaultBaseUrl/acceptsApiKey/requiresApiKey/requiresBaseUrl/routerSupported/deployment/blocking/canProbe/readmeScenarios/whatYouNeed/defaultDirectModel/defaultModel/presets/capabilities/fields[]
- **channels**:type/label/description/transport/requiresPublicUrl/dependencyExtra/restartRequired/docsHint/fields[]/help/blocking/canProbe/readmeScenarios/setupAids[]
- **searchProviders**:providerId/label/runtimeSupported/metadataSupported/requiresApiKey/envKey/deployment/blocking/canProbe/readmeScenarios/whatYouNeed/capabilities/fields[]
- **routerProfiles**:defaultTier/textTiers/modes[]/profiles[](每个 profile:providerId/label/tierProfile/description/tierMappings/tags)
- **memoryEmbeddingProviders / imageGenerationProviders / audioProviders**:类似 provider,含 defaultModel/defaultTtsVoice/defaultLanguageCode 等

CLI 侧同构:`onboarding/setup_engine.py:61` 的 `setup_catalog_payload()`,支持按 section 别名过滤。

### 2.3 配置持久化(config_store.py)

- **路径解析**(config_store.py:106 `resolve_config_path`):显式路径 → `OPENSQUILLA_GATEWAY_CONFIG_PATH` → `cwd/opensquilla.toml` → `~/.opensquilla/config.toml`
- **格式**:TOML(tomli/tomli_w 读写)
- **加载**(config_store.py:203 `load_config`):读 TOML → `migrate_config_payload` 迁移 → `GatewayConfig.model_validate` → `_mark_env_absorbed_runtime_secrets` → 记 `_persist_baseline` 用于 diff
- **持久化**(config_store.py:874 `persist_config`):diff-based——只把当前模型与基线的差异写入 TOML;环境变量来源的值不写文件;原子写(临时文件 + rename) + 0600 权限
- **provider key 存储**:`[llm]` section 的 `api_key` 字段;profile 密钥存 `[llm_profiles.<id>]`;环境变量来源标记 runtime secret 不落盘
- **与 gateway/config.py 关系**:`GatewayConfig`/`LlmProviderConfig`/`LlmProviderProfile` 等 pydantic 模型是配置单一事实来源;运行时 `ctx.config` 就是 `GatewayConfig` 实例;`config_secrets.py` 的 `inherit_runtime_secrets()` 负责突变后同步密钥标记

### 2.4 探针流程(probe.py + probe_history.py)

- `probe_llm_provider()`(probe.py:59):`(provider_id, model, api_key, api_key_env, base_url, proxy, allow_default_api_key_env, chat_stream_factory) -> ProviderProbeResult`;`ProviderProbeResult.to_payload()` → ok/providerId/model/failureKind/message/code/latencyMs/firstResponseMs/totalMs
- 执行:用 `build_provider` 构造临时 provider → 发一个 one-token chat 请求 → 30s 超时(`_PROBE_TIMEOUT_SECONDS`);build 失败(缺 key)直接 ok=False 不进网络
- `discover_selectable_provider_models()`:模型列表发现,只对 registry 验证过的 provider + 官方 host 查询,fail-closed
- `discover_image_generation_models()`(image_generation_model_discovery.py):只用官方 image-model 端点
- **probe_history**(probe_history.py):文件 `{state_dir}/onboarding/probe_history.json`,每条含 ok/failure_kind/timestamp/fingerprint(SHA-256:api_key digest + apiKeyEnv + baseUrl + proxy);`saved_deployment_fingerprint()` 判断"凭证变了没";`load_probe_history()` / `record_probe()`

### 2.5 状态机与 section 校验

- **section_status.py**:`SectionStatus` 枚举(OK/MISSING/DEGRADED/OPTIONAL/UNKNOWN,section_status.py:56);`section_verifiers()`(section_status.py:525)注册 8 个 section verifier(llm/router/ensemble/search/channels/image_generation/audio/memory_embedding);`needs_onboarding()`(section_status.py:539)中 `FIRST_RUN_REQUIRED_SECTIONS={"llm"}`——只有 LLM 阻塞首次运行;verifier 是纯函数 `(cfg: GatewayConfig) -> SectionStatus`,从不抛异常
- **status.py**:`OnboardingStatus` 数据类(status.py:61-98);`get_onboarding_status()`(status.py:1000)依次调 8 个 verifier → 组装
- **setup_engine.py**:`SetupEngine`(setup_engine.py:99)持有 config 实例,`status()/catalog()/apply(section, payload)/persist()`;apply 按 section 名分派到对应 `upsert_*` mutation
- **flow.py**:`run_interactive_onboard()`(flow.py:2788)交互式向导;`--if-needed` → banner → `_ask_existing_setup_action` → `_run_onboard_walk`(provider → router → ensemble → search → channels → image-generation → audio → memory-embedding)

### 2.6 与 gateway 其他服务的对接面(写方法调用的副作用)

1. `gateway/llm_runtime.py`:resolve_llm_runtime_config/resolve_llm_credential/discard_profile_credential_pool
2. `gateway/config_secrets.py`:inherit_runtime_secrets
3. `gateway/model_routing.py`:broadcast_model_routing_changed
4. `gateway/model_catalog_refresh.py`:refresh_live_model_catalog/reconcile_tokenrhythm_profile_transition
5. `gateway/channels_bridge.py`:get_channels_reconciler(热重载)
6. `gateway/config.py`:pydantic 模型
7. `provider/selector.py`:sync_primary
8. `provider/registry.py`:get_provider_spec
9. `provider/ensemble.py`:ensemble_runtime_status
10. `tools/builtin/media.py`:configure_image_generation/configure_audio
11. `tools/builtin/web.py`:configure_search
12. `engine/selector_override.py`:acquire_profile_credential/report_profile_credential_failure
13. `engine/usage_accounting.py`:account_provider_stream/bind_usage_accounting_scope

**关键事务顺序**:所有写方法遵循 `persist 优先 → apply_inplace → sync 副作用`——先写磁盘,成功后才更新运行时与工具,保证写入失败时运行时不被污染。

---

## 3. Rust 现状测绘(缺口事实)

### 3.1 `crates/gateway/src/onboarding.rs` —— 死代码

`register_onboarding_handlers()`(onboarding.rs:78)注册 12 个 handler,与前端要的 camelCase 完全不同(是另一套极简 wizard API):

| Rust RPC 方法名 | 参数 | 返回形状 |
|---|---|---|
| `onboarding.providers` | 无 | `{"providers": [ProviderSpec], "count": usize}` |
| `onboarding.start` | 无 | `{"started": true, "state", "progress"}` |
| `onboarding.status` | 无 | `OnboardingStatusResponse`(state/progress/steps/is_complete/is_cancelled) |
| `onboarding.select_provider` | provider | `{"selected"}` |
| `onboarding.set_api_key` | api_key | `{"status": "set"}` |
| `onboarding.set_model` | model | `{"model"}` |
| `onboarding.advance` / `back` / `skip` | 无 | `{"state"}` |
| `onboarding.complete` | 无 | `{"completed": true}` |
| `onboarding.cancel` | 无 | `{"cancelled": true}` |
| `onboarding.save` | 无 | `{"saved": true}` |

关键问题:
- **未注册**:`Gateway::new`(app.rs:93-163)与 `register_domain_handlers`(rpc_handlers.rs:97-111)均未调用 `register_onboarding_handlers`。grep 确认 app.rs 无任何 onboarding/wizard 引用。`onboarding.start`/`save`/`complete` 操作的是 `SetupFlow` 内部的独立 `Config` 副本 + 独立 `ConfigStorage`,与 `AppState.config` / `AppState.config_store` 完全隔离——即使注册也无法反映真实配置。
- 引用的类型:`OnboardingSession`(onboarding.rs:20-46,持 `Arc<Mutex<Option<SetupFlow>>>`)、`OnboardingStatusResponse`(:55-62,有 serde)、`SetupFlow`(来自 `crates/onboarding/src/flow.rs:48-317`,9 态状态机 Welcome→…→Complete/Cancelled)、`ProviderSpec`(来自 `crates/onboarding/src/providers.rs:4-207`,`discover_providers()` 硬编码 OpenAI/Anthropic/DeepSeek/Ollama/OpenRouter 5 家)、`ConfigStorage`(来自 `crates/onboarding/src/storage.rs:28-181`,YAML/JSON 扁平 KV 文件存储)。
- `crates/onboarding/` crate 存在,共 5 文件:channels.rs / lib.rs / providers.rs / flow.rs / storage.rs——是另一条"向导"实现线,与 Python 的配置即事实来源模型脱节。

### 3.2 注册缺口

`Gateway::new` 注册的命名空间(按序):sessions.* → chat.* → config.* → cron.* → system.* → channels.* → usage.* → approvals.* + `register_extra_rpc`(app.rs:106-130)。`wizard.*` 同样未注册。onboarding 补注册点即 app.rs:106-130 之间(或 `GatewayBuilder::handler()`)。

### 3.3 src-tauri 命令面(agent_bridge.rs)

| Tauri 命令 | 位置 | 功能 | 持久化 |
|---|---|---|---|
| `list_providers` | :1020 | 读 state.config.providers | 只读 |
| `list_models` | :1034 | 聚合 providers[].models | 只读 |
| `get_provider_status` | :1358 | 单 provider(name/providerType/configured/models/baseUrl) | 只读 |
| `get_all_provider_statuses` | :1377 | 全部 provider 状态 | 只读 |
| `get_config` | :1173 | 读 state.config 完整 JSON | 只读 |
| `get_config_effective` | :1397 | 合并 state.config + config_store.overrides() | 只读 |
| `set_config` | :1181 | 写 state.config + 显式 `config.save()`(agent_bridge.rs:1204) | **落盘** |
| `patch_config` | :1411 | `state.config_store.patch()`(内存 overlay) | **不落盘** |
| `reset_config` | :1442 | config_store.reset()/delete() | **不落盘** |

结论:provider 读命令齐全,**零写命令**(无 add_provider/update_provider/set_api_key 之类);唯一写入口 `set_config`(整体替换)与 `patch_config`(内存 overlay)。

### 3.4 持久化缺陷 —— ConfigStore 与 AppState.config 两套对象

- `AppState`(src-tauri/src/state.rs:24-46):`config: Arc<RwLock<Config>>`(:31,来自 `Config::load()` 磁盘文件)+ `config_store: Arc<opensquilla_gateway::ConfigStore>`(:41)
- `AppState::new`(state.rs:65-81):config 直接包裹(:73);`config_store: Arc::new(opensquilla_gateway::ConfigStore::new())`(:77)——**全新空实例**,内部 `Config::default()` + 空 HashMap overlay
- `ConfigStore` 结构(crates/gateway/src/config.rs:24-28):`entries: Arc<Mutex<HashMap<String, Value>>>`(内存 overlay)+ `config: Arc<Mutex<Config>>`(内部副本)+ `router: Arc<RwLock<ModelRouter>>`
- **patch 不落盘证据**:`ConfigStore::patch`(config.rs:113-121)只写 `self.entries`,无 `config.save()`/文件 I/O;`ConfigStore::save`(config.rs:253-261)存在但 patch 不调用;`patch_config`(agent_bridge.rs:1411-1439)只调 `config_store.patch`——**进程重启即丢**
- 两套对象之间无任何同步机制:`AppState.config` 与 `AppState.config_store` 完全隔离

---

## 4. 前端调用面测绘

### 4.1 注册表绑定

`TAURI_METHOD_REGISTRY`(opensquilla-webui/src/lib/rpc.ts:864-1069)共 52 条,分属 chat/sessions/config/providers/models/skills/system-desktop/gateway-lifecycle 七类,**0 条 onboarding.\***。与 onboarding 相关的仅:providers.list(:1021)→ list_providers、providers.status(:1025)→ get_all_provider_statuses、models.list(:1034)→ list_models、config.get(:984)/config.effective(:988)/config.patch(:996)/config.patch.safe(:1003)→ 对应命令。grep 确认 rpc.ts 全文无 `onboarding.`。

### 4.2 前端 27 个 onboarding.* 调用点

**useSetupCatalog.ts(设置/向导主界面)**
- `onboarding.catalog`:736;`onboarding.status`:737
- `onboarding.provider.configure`:3385(saveProvider 降级分支)、3577(applyProviderPreset)
- `onboarding.provider.probe`:2442;`onboarding.provider.credential.reveal`:2633;`onboarding.provider.credential.clear`:2667
- `onboarding.models.discover`:633、647(两处 fallback)
- `onboarding.llmProfile.upsert`:3370;`onboarding.llmProfile.activate`:2496、3533;`onboarding.llmProfile.active.remove`:2789;`onboarding.llmProfile.remove`:2797;`onboarding.llmProfile.probe`:2442;`onboarding.llmProfile.models.discover`:621;`onboarding.llmProfile.credential.clear`:2668
- `onboarding.router.configure`:3460、3511;`onboarding.ensemble.configure`:3477、3519
- `onboarding.capability.reset`:3166;`onboarding.imageGeneration.models.discover`:558
- `onboarding.search.configure`:3597;`onboarding.memory_embedding.configure`:3613;`onboarding.imageGeneration.configure`:3631;`onboarding.audio.configure`:3650

**useSetupProviderForm.ts**:`onboarding.provider.probe`:709;`onboarding.models.discover`:787;`onboarding.llmProfile.probe`:709;`onboarding.llmProfile.models.discover`:786;`onboarding.llmProfile.draft.probe`:708;`onboarding.llmProfile.draft.models.discover`:784

**channelRpc.ts**:`onboarding.channel.probe`:54;`onboarding.channel.upsert`:64
**ChannelsView.vue**:`onboarding.channel.remove`:1454;`onboarding.channel.enable`/`disable`:1730
**ChatView.vue**:`onboarding.status`:1475(经 useRpcCall,只读 audioConfigured)

### 4.3 supportsMethod / profileSaveSupported 推导链

1. `TauriRpcClient.connect()`(rpc.ts:1394-1403)dispatch `_hello`,带 `features.methods = TAURI_SUPPORTED_METHODS`(rpc.ts:1076-1078,即 registry 键集合)
2. RPC store `_hello` 处理器(stores/rpc.ts:137-148)写入 `methods` reactive 数组
3. `supportsMethod(method)`(stores/rpc.ts:203-205)= `methods.includes(method) && !unavailableMethods.has(method)` → onboarding.* 恒 false
4. `profileSaveSupported`(useSetupCatalog.ts:1692-1695)= `typeof rpc.supportsMethod !== 'function' || rpc.supportsMethod('onboarding.llmProfile.upsert')` → **恒 false**;`primaryProviderRemovalSupported`(rpc.ts:1696-1699)同样 false
5. `rpc.call` 的 METHOD_NOT_FOUND 路径(rpc.ts:1418-1425):binding 不存在直接 `throw new TauriRpcError('METHOD_NOT_FOUND')` → `normalizeTauriError`(rpc.ts:1284-1300)统一映射

### 4.4 降级路径

1. **saveProvider 降级**(useSetupCatalog.ts:3323-3409):profileSaveSupported=false → `replacesPrimaryOnLegacyGateway`=true → 跳过 `llmProfile.upsert`(3370)走 `provider.configure`(3385)——**该路径同样未注册,降级也失败**
2. **supportsMethod 守卫静默跳过**(useSetupCatalog.ts:540-544):`onboarding.imageGeneration.models.discover` 不 supported 时 `return Promise.resolve()`,imageModelCatalogs 回退到 `curatedImageModelCatalog`(spec 的 suggestedModels/defaultModel)
3. **tierModelDiscovery 隐式降级**:`discoverTierProviderModels()`(rpc.ts:592-677)无显式 check,但 `llmProfile.models.discover`/`models.discover` 抛 METHOD_NOT_FOUND 被 catch(rpc.ts:662-668)吞掉,tierModelCatalogs={models:[], source:'none'}
4. **loadData 整体失败**(rpc.ts:733-808):`Promise.all([... onboarding.catalog, onboarding.status ...])` 在 Tauri 下整体 reject → 外层 catch(rpc.ts:804-807)toast **"Failed to load settings"**,设置界面无法加载
5. **ChannelsView 全部 channel.\* 失败**:probe/upsert/remove/enable/disable 全部 METHOD_NOT_FOUND,渠道操作不可用
6. **ChatView mic 静默失败**:`useRpcCall` 内部 catch,voiceReady 恒 false

### 4.5 前端期望的 catalog 结构

`OnboardingCatalog` 接口(useSetupCatalog.ts:278-287):`providers? / routerProfiles?(profiles:[{providerId,tiers?}], defaultTier) / searchProviders? / imageGenerationProviders? / memoryEmbeddingProviders?`。`ProviderSpec`(useSetupCatalog.ts:112-128)字段:providerId/label/runtimeSupported/routerSupported/fields/whatYouNeed/envKey/acceptsApiKey/requiresApiKey/defaultBaseUrl/defaultDirectModel/defaultModel/suggestedModels/deployment/presets。

消费:runtimeProviders=catalog.providers.filter(runtimeSupported)(rpc.ts:836)驱动 provider 选择器;routerProfiles 驱动路由预设(rpc.ts:1030);search/image/memory 各自 filter;providerSpec 从 runtimeProviders 匹配当前选中(rpc.ts:1062-1064)。

---

## 5. 方法级缺口矩阵

> 列:方法名 / Python 位置 / Rust 现状 / 前端调用点 / 打通难度

| 方法 | Python | Rust | 前端调用 | 难度 |
|---|---|---|---|---|
| onboarding.status | rpc_onboarding.py:412 | 无(现有简化版不同构) | useSetupCatalog:737, ChatView:1475 | 中(status 需 8 个 section verifier) |
| onboarding.catalog | :417 | 无 | useSetupCatalog:736 | 中(7 spec 聚合) |
| provider.configure | :579 | 无 | useSetupCatalog:3385,3577 | 高(写 + persist + sync) |
| provider.probe | :1187 | 无 | useSetupProviderForm:709, useSetupCatalog:2442 | 高(真实 LLM 探测) |
| provider.credential.reveal | :1253 | 无 | useSetupCatalog:2633 | 中(密钥读,需安全策略) |
| provider.credential.clear | :1259 | 无 | useSetupCatalog:2667 | 中 |
| models.discover | :1307 | 无 | useSetupProviderForm:787, useSetupCatalog:633,647 | 高(模型列表发现) |
| llmProfile.upsert | :629 | 无 | useSetupCatalog:3370 | 高 |
| llmProfile.credential.clear | :681 | 无 | useSetupCatalog:2668 | 中 |
| llmProfile.remove | :723 | 无 | useSetupCatalog:2797 | 中 |
| llmProfile.active.remove | :747 | 无 | useSetupCatalog:2789 | 高(需 replacement 联动) |
| llmProfile.activate | :833 | 无 | useSetupCatalog:2496,3533 | 高 |
| llmProfile.probe | :1043 | 无 | useSetupProviderForm:709, useSetupCatalog:2442 | 高 |
| llmProfile.draft.probe | :1080 | 无 | useSetupProviderForm:708 | 高 |
| llmProfile.models.discover | :1114 | 无 | useSetupProviderForm:786, useSetupCatalog:621 | 高 |
| llmProfile.draft.models.discover | :1151 | 无 | useSetupProviderForm:784 | 高 |
| router.catalog | :1392 | 无 | (前端走 onboarding.catalog) | 中 |
| router.configure | :1399 | 无 | useSetupCatalog:3460,3511 | 高(路由层) |
| ensemble.configure | :1439 | 无 | useSetupCatalog:3477,3519 | 高 |
| channel.probe | :1478 | 无 | channelRpc:54 | 高(需渠道探针) |
| channel.upsert | :1802 | 无 | channelRpc:64 | 高(liveApply) |
| channel.remove | :1827 | 无 | ChannelsView:1454 | 中 |
| channel.enable | :1869 | 无 | ChannelsView:1730 | 中 |
| channel.disable | :1874 | 无 | ChannelsView:1730 | 中 |
| search.configure | :1510 | 无 | useSetupCatalog:3597 | 高 |
| imageGeneration.configure | :1544 | 无 | useSetupCatalog:3631 | 高 |
| imageGeneration.models.discover | :1371 | 无 | useSetupCatalog:558 | 中(有前端 curated fallback) |
| memory_embedding.configure | :1586 | 无 | useSetupCatalog:3613 | 高 |
| audio.configure | :1703 | 无 | useSetupCatalog:3650 | 高 |
| capability.reset | :1720 | 无 | useSetupCatalog:3166 | 中 |

全部 30 项 Rust 现状 = "无"。现有 `crates/gateway/src/onboarding.rs` 的 12 个方法不是这些方法的近似,是一套独立简版,不可复用(需重写而非桥接)。

---

## 6. 修复路径建议

按"先救设置界面 → 再打通写路径 → 最后补齐探测/发现"排序:

1. **P0 持久化层修好(前置依赖)**:统一 `ConfigStore` 与 `AppState.config`。两条路二选一:
   - 让 `patch_config` 落盘:合并 overlay 进 `AppState.config` 的 RwLock 并调 `Config::save()`(对齐 Python `persist_config` 的 diff + 原子写语义);
   - 或废弃 `ConfigStore.entries` overlay 模型,让 ConfigStore 直接持真实 Config。**必须**:任一 write 命令写后进程重启不丢。
2. **P0 最小救场(可选快速回退)**:前端 `loadData()` 的 `onboarding.catalog`/`onboarding.status` 失败导致整个设置界面挂。可在 Rust 侧先实现 `onboarding.catalog` + `onboarding.status` 两个只读方法并注册(基于现有 provider registry 只读命令 + section verifier),先让界面能打开,再逐方法补写。
3. **P1 写方法面**:按依赖顺序移植 `provider.configure` → `llmProfile.upsert/activate` → `router.configure` → `ensemble.configure` → `search/imageGeneration/memory_embedding/audio.configure` → `capability.reset` → `channel.upsert/remove/enable/disable`。每个写方法严格遵循 Python 的 `persist 优先 → apply_inplace → sync 副作用` 顺序。
4. **P2 probe / discover**:`provider.probe`/`llmProfile.probe`/`draft.probe`(真实 LLM one-token 探测)、`models.discover` 系列(模型列表发现,fail-closed)、`provider.credential.reveal`(需安全策略)。probe_history 指纹机制可后续跟进。
5. **P3 前端注册表**:补齐后把新方法加进 `TAURI_METHOD_REGISTRY`(每条必须带 `command` 字段,否则 TS2741),`supportsMethod`/`profileSaveSupported` 自动恢复 true,旧降级分支自然失效。
6. **死代码处理**:`crates/gateway/src/onboarding.rs` + `crates/onboarding/` 现实现与目标方法面不符,建议明确废弃或重写为薄 RPC 层(复用其 serde 结构而非逻辑)。

**架构提醒**:Python 的写方法调用 13 个 gateway 服务做副作用同步(llm_runtime/config_secrets/model_routing/model_catalog_refresh/channels_bridge/selector/ensemble/tools.builtin.media/web/selector_override/usage_accounting)。Rust 移植时需逐一确认每个副作用在 Rust 网关是否有对应物(如 `broadcast_model_routing_changed`、`refresh_live_model_catalog`、channels hot-reload),没有的标为文档化偏差。

---

## 7. 约束与验证方式

- 本次调研为**纯只读**:只用了 Read/Glob/Grep,未运行任何 cargo/npm/python(遵循"禁止本地构建安装依赖静态命令检查")。
- 承重事实已由主代理复核:`register_onboarding_handlers` 仅存在于 onboarding.rs 内部(未在 app.rs 调用)、`patch_config`(agent_bridge.rs:1411)→`config_store.patch`(config.rs:113-121)无落盘、rpc.ts 无任何 onboarding.* 条目。
- 实施阶段的验证只能靠静态复核 + gh CLI 触发远程 CI(本仓库契约断裂 CI 抓不到,需人工/审计按 §5 矩阵四维对齐:方法名/参数形状/返回形状/事件通道)。
