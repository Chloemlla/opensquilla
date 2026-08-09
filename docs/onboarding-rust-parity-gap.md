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

---

# Part 2 — 同类缺口审计:前端调用但 Rust 未绑定/未实现的方法面

> 状态:调研完成(2026-08-09,3 个并行只读子代理:Python 参照面 / Rust 网关面 / 前端调用面,主代理复核)
> 范围:全应用,不只 onboarding。与 onboarding 同属"前端 rpc.call 一个方法,但 Rust 侧没有或没接通"这一大类。

## P1. 结构性根因:两层绑定缺口

一个方法面要"通",需要**两层都通**,当前只有约 30% 的方法面两层都通:

| 层 | 载体 | 现状 |
|---|---|---|
| L1 网关 RPC(WS/dev 模式) | `Gateway::new`(`crates/gateway/src/app.rs:106-131`)把 `register_*_handlers` 挂到 `RpcRegistry` | 只注册 9 组 handler;`crates/gateway/src/` 下有 **20 个** `register_*_handlers`,其中 **11 个从未被调用**(死代码) |
| L2 Tauri 命令(桌面模式) | `src-tauri/src/main.rs:143-210` 的 `invoke_handler` + 前端 `TAURI_METHOD_REGISTRY`(`opensquilla-webui/src/lib/rpc.ts:864-1069`) | `invoke_handler` 注册 41 个命令,只覆盖网关面约 30%;`TAURI_METHOD_REGISTRY` 52 条,0 条 onboarding,且大量已注册网关方法无绑定 |

**结论**:即便 Rust 网关已实现且注册(如 `channels.*`、`cron.*`、`usage.*`、`sessions.subscribe/reset`),只要 L2 没绑定,前端在 Tauri 桌面模式下依旧 `METHOD_NOT_FOUND`。修复不能只"在 Rust 里实现",必须一路打通到 L2(或让前端统一走网关 RPC)。

## P2. Rust 网关 handler 现状

`Gateway::new` 实际注册的 9 组(app.rs:106-131):

```
sessions.* / chat.* / config.* / cron.* / system.* / channels.* / usage.* / approvals.*
+ register_extra_rpc(crates/gateway/src/http_api.rs:689-886)
  → channels.status / channels.logout / channels.pairings / channels.pairing.approve /
    channels.pairing.revoke / usage.status / usage.cost / exec.approvals.get|set /
    exec.approval.resolve / system.shutdown
```

**已实现但从未注册的 11 个 handler 模块(死代码,与 onboarding.rs 同型)**:`agents.rs`、`workspaces.rs`、`sandbox.rs`、`skills.rs`、`proposals.rs`、`migration.rs`、`routing.rs`、`meta_runs.rs`、`models.rs`、`doctor`、`memory`——均有完整实现,但 `Gateway::new` 一个都没调用。

## P3. 缺口分类

### 类 A — Rust 网关已注册,只差 L2(Tauri 命令 + 前端注册表)绑定(7 组)

| 命名空间 | 网关已注册方法 | 关键位置 | 备注 |
|---|---|---|---|
| `channels.*` | create/get/list/update/delete/send/register_handle/list_handles/status/logout/pairings/pairing.approve/pairing.revoke | `channels.rs:98-350` + `http_api.rs:689` | `pairing.revoke` Python 版还同步 admin 移除,Rust 只 `enabled=false` |
| `cron.*` | create/get/list/update/delete/pause/resume/disable/stats/executions/due/run/runs/subscribe | `cron.rs:124-440` | 已正确接线 scheduler crate(少数例外) |
| `usage.*` | record/summary/list/by_model/by_session/clear/status/cost | `usage.rs:159` + `http_api.rs:789` | 无 `usage.query` 名,`summary`+`list` 可覆盖 |
| `sessions.subscribe/reset` | subscribe(:1513)/reset(:1421) | `sessions.rs` | subscribe 是 long-poll 实现;steer v1 返回 501 |
| `config.routing.*` | list/set_rule/override/release/decisions/route | `config.rs:539-681` | 前端要的是 `models.routing.*`,命名不同 |
| `chat.*` | 15 个 | `chat.rs:411` | 前端 `chat.*` 已绑,部分冗余 |
| `approvals.*` | 7 个 | `approvals.rs:105` | — |

**修复成本最低**:只在 L2 加命令 + 注册表条目,不改网关逻辑。

### 类 B — Rust 已实现但从未注册(死代码,8 组),需"接线 + L2 绑定"

| 命名空间 | 实现文件 | 方法数 | 前端在用 |
|---|---|---|---|
| `agents.*` | `agents.rs:83-280` | 6 | agents.create/update/delete(AgentsView.vue) |
| `workspaces.*` | `workspaces.rs:81-300` | 8 | list/open/update/remove/pin/history.delete |
| `sandbox.*` | `sandbox.rs:87-290` | 8 | 与 Python **方法面正交**,见 §P5 |
| `skills.*` | `skills.rs:117-320` | 9 | skills.get(Tauri 只有 list_skills) |
| `proposals`(exec.proposals.*) | `proposals.rs:117-280` | 6 | 前端 plans.* **不是** proposals,见 §P5 |
| `migration.*` | `migration.rs:96-210` | 4 | migration.sources.*,命名不同 |
| `routing.*`(routing.hold 等) | `routing.rs:95-280` | 6 | 前端要 models.routing.*,命名不同 |
| `meta_runs.*` | `meta_runs.rs:141-350` | 7 | meta.runs.confirm_preflight/replay |

### 类 C — Rust 完全没有,需从 Python 移植

- `plans.*` 全部(5 个方法,在 `rpc_sessions.py:7189-7696`,是内嵌命名空间)
- `sandbox` 用户面全部:setup.status/setup.ensure/resume/explain/run_mode.*/run_context.*/mount.*/domain.*/bundle.*/path.*/workspace.set
- `channels.admin.set`(rpc_channels.py:775)、`channels.restart`(:530)、`channels.probe`(:440)
- `sessions.unsubscribe`、`sessions.messages.subscribe/unsubscribe`、`sessions.steer.v2`(:4182)
- `models.routing.get/set`(:258/:268)
- `usage.query`(:727,时间窗口聚合)
- `workspaces.pin`(:158)、`workspaces.history.delete`(:210)
- `skills.status/reload/install/update/uninstall/deps.install/bins`

### 类 D — 命名差异(别名未注册,近乎零成本)

| 前端/ Python 调用 | 现有 Rust 等价 | 说明 |
|---|---|---|
| `cron.status` | `cron.get` | Python 别名 |
| `cron.add` | `cron.create` | Python 别名 |
| `cron.remove` | `cron.delete` | Python 别名(前端 cron.remove 会打 L2 缺绑定) |
| `cron.runs` | `cron.executions` | Python 别名 |
| `workspaces.remove` | `workspaces.delete` | 语义等价 |
| `migration.sources.list` | `migration.discover` | 命名空间不同 |
| `migration.sources.preview` | `migration.preview` | 命名空间不同 |
| `models.routing.set` | `config.routing.set_rule` | 跨命名空间,需适配层 |
| `usage.query` | `usage.summary`+`usage.list`+`usage.by_model`+`usage.by_session` | 需聚合层 |

## P4. 前端调用面与降级地图(约 40+ 方法面 / 50+ 调用点 / 13 个功能域)

降级路径三型:①`supportsMethod` 守卫(未支持则静默隐藏/跳过)→ 守卫位置;②`try/catch` 吞掉(降级本地缓存/空数据)→ catch 行为;③硬失败(toast/功能不可用)。

| 方法面 | 降级类型 | 影响功能 | 关键位置 |
|---|---|---|---|
| `agents.create/update/delete` | ③ toast | Agents 增删改全失效 | AgentsView.vue:396,433,466 |
| `channels.pairings` | ③ toast | 配对列表无法加载 | useChannelMembers.ts:180 |
| `channels.pairing.approve/revoke` | ③ toast | 审批/撤销无响应 | :209,:309 + ChannelsView.vue:992 |
| `channels.admin.set` | ③ toast | 管理员设置无响应 | useChannelMembers.ts:252,287 |
| `channels.restart` / `channels.probe` | ③ toast | 重启/探测无响应 | ChannelsView.vue:1717,1687 |
| `onboarding.channel.enable/disable` | ③ toast | 启用/禁用开关无响应 | ChannelsView.vue:1730 |
| `cron.list` | ③ useRequest | Cron 列表空白 | useCronJobs.ts:26 |
| `cron.run` / `cron.runs` | ③ toast | 手动触发/历史空白 | :147 / useCronRuns.ts:13 |
| `cron.create/update/remove` | ③ toast/未捕获 | 增删改全失效 | useCronForm.ts:276, useCronJobs.ts:132,167 |
| `sandbox.setup.status` | ② 静默轮询 | 沙箱状态永远 pending | useSandboxSetupRecovery.ts:78,91-97 |
| `sandbox.setup.ensure` | ③ error | 设置按钮无效 | :110,113-116 |
| `sandbox.resume` | ③ toast | Resume 无响应 | ChatView.vue:3038 |
| `workspaces.list` | ① 守卫空列表 | 工作区列表永不显示 | stores/rpc.ts:110-112 |
| `workspaces.open/remove/update/pin/history.delete` | ③ throw | 工作区全部操作不可用 | useProjectWorkspaces.ts:98-180 |
| `skills.get` | ③ error | 详情面板永远报错 | useSkillDetailController.ts:64,88-89 |
| `plans.setMode/implement/revise/cancelRun` | ① 守卫隐藏 UI + ③ | Plan 模式完全不可见/不可用 | useChatPlans.ts:391-531, ChatView.vue:2478-2480 |
| `migration.sources.list/preview` | ① 守卫 unsupported | 迁移面板不可用 | DataMigrationPanel.vue:529,588 |
| `sessions.subscribe/unsubscribe` | ② warn | 会话列表不自动刷新 | useSessionListSubscription.ts:63,117 |
| `sessions.reset` | ② warn | `/reset` 斜杠命令静默失效 | useChatSlashCommands.ts:385,389 |
| `sessions.contextCompact` | ③ toast | `/compact` 斜杠命令失效 | :400 |
| `models.routing.get` | ② catch + config 投影 | 降级只读兼容(**设计允许**) | rpc.ts:1031-1033, useChatFeatureToggles.ts:170,176 |
| `models.routing.set` | ③ toast + 回滚 | 路由模式切换失效 | useChatFeatureToggles.ts:262 |
| `usage.query` | ② → `usage.status` 回退 | Usage 页面降级(有缓存链) | useUsageQuery.ts:539-576 |
| `usage.status` | ② 保留缓存 | 缓存快照保留 | :585-589 |
| `meta.runs.confirm_preflight` | ③ error | 预审批无响应 | useMetaRuns.ts:186 |
| `meta.runs.replay` | ③ toast | 重放/重试无响应 | :276 |
| `onboarding.*`(27 个) | ③ toast + 守卫 | 设置向导全流程不可用 | useSetupCatalog.ts 等(见 Part 1) |

> 前端只有 `models.routing.get` 与 `usage.query` 有完整 fallback 链;其余均为硬失败或守卫隐藏。`cron.status/cron.add` 前端并不调用(前端实测只有 list/run/runs/create/update/remove)。

## P5. 方法级缺口矩阵(合并三面)

> 列:前端方法面 / Python 实现(rpc_*.py:行)/ Rust 网关状态 / L2 Tauri 绑定 / 分类

### agents.*
| 前端 | Python | Rust | L2 | 类 |
|---|---|---|---|---|
| agents.create | rpc_agents.py:192 | 已实现未注册(agents.rs:83) | 无 | B |
| agents.update | :230 | 已实现未注册 | 无 | B |
| agents.delete | :263 | 已实现未注册 | 无 | B |

### channels.*
| 前端 | Python | Rust | L2 | 类 |
|---|---|---|---|---|
| channels.pairings | rpc_channels.py:550 | 已注册(http_api.rs:722) | 无 | A |
| channels.pairing.approve | :653 | 已注册(http_api.rs:749) | 无 | A |
| channels.pairing.revoke | :819 | 已注册(http_api.rs:769,只 enabled=false) | 无 | A |
| channels.admin.set | :775 | 无 | 无 | C |
| channels.restart | :530 | 无 | 无 | C |
| channels.probe | :440 | 无 | 无 | C |

### cron.*
| 前端 | Python | Rust | L2 | 类 |
|---|---|---|---|---|
| cron.list | rpc_cron.py:556 | 已注册(cron.rs) | 无 | A |
| cron.create | :580(别名) | 已注册 cron.create | 无 | A(别名 D) |
| cron.update | :738 | 已注册(cron.rs:233) | 无 | A |
| cron.remove | :965(别名 cron.delete) | 已注册 cron.delete | 无 | A(别名 D) |
| cron.run | :974 | 已注册 | 无 | A |
| cron.runs | :983(别名 executions) | 已注册(cron.rs:407) | 无 | A(别名 D) |

### sandbox.*(Python 与 Rust 方法面**正交**)
| 前端 | Python | Rust | L2 | 类 |
|---|---|---|---|---|
| sandbox.setup.status | rpc_sandbox.py:670 | 无(只有内部策略方法) | 无 | C |
| sandbox.setup.ensure | :676 | 无 | 无 | C |
| sandbox.resume | :719 | 无 | 无 | C |

> Python sandbox 是用户配置面(setup/mount/domain/bundle/path/run_mode/run_context/workspace.set);Rust sandbox.rs 是内部策略面(select_level/build_policy/get_policy/list_contexts/remove_context/preview/record_result/get_result)。**没有一个方法名重叠**——Rust 侧的 sandbox 实现无法直接对接前端,需整体移植 Python 用户面。

### workspaces.*
| 前端 | Python | Rust | L2 | 类 |
|---|---|---|---|---|
| workspaces.list | rpc_workspaces.py:91 | 已实现未注册(workspaces.rs:155) | 无 | B |
| workspaces.open | :105 | 已实现未注册(:170) | 无 | B |
| workspaces.update | :136 | 已实现未注册(:219) | 无 | B |
| workspaces.remove | :177 | 有 delete(workspaces.rs:268)未注册 | 无 | B(命名 D) |
| workspaces.pin | :158 | 无 | 无 | C |
| workspaces.history.delete | :210 | 无 | 无 | C |

### skills.*
| 前端 | Python | Rust | L2 | 类 |
|---|---|---|---|---|
| skills.list | rpc_skills.py:396 | 已实现未注册(skills.rs:167) | **有**(list_skills) | 已通(仅 list) |
| skills.get | :449 | 已实现未注册(skills.rs:191) | 无 | B |
| skills.status/reload/install/update/uninstall/deps.install/bins | :366-651 | 无 | 无 | C |

### plans.*(≠ proposals!内嵌在 rpc_sessions.py)
| 前端 | Python | Rust | L2 | 类 |
|---|---|---|---|---|
| plans.capabilities | rpc_sessions.py:7189 | 无 | 无 | C |
| plans.setMode | :7203 | 无 | 无 | C |
| plans.implement | :7309 | 无 | 无 | C |
| plans.revise | :7478 | 无 | 无 | C |
| plans.cancelRun | :7555 | 无 | 无 | C |

### migration.sources.*
| 前端 | Python | Rust | L2 | 类 |
|---|---|---|---|---|
| migration.sources.list | rpc_migration.py:415 | 有 discover(migration.rs:98)未注册 | 无 | B(命名 D) |
| migration.sources.preview | :460 | 有 preview(:109)未注册 | 无 | B(命名 D) |

### sessions.*
| 前端 | Python | Rust | L2 | 类 |
|---|---|---|---|---|
| sessions.subscribe | rpc_sessions.py:6788 | 已注册(sessions.rs:1513,long-poll) | 无 | A |
| sessions.unsubscribe | :6796 | 底层有但未注册 | 无 | C |
| sessions.reset | :5330 | 已注册(sessions.rs:1421) | 无 | A |
| sessions.contextCompact | (hydrate/snapshot 对应) | hydrate/snapshot 已注册 | 无 | A |
| sessions.steer.v2 | :4182 | 只有 v1(sessions.rs:1543,返回 501) | 无 | C |

### models.routing.*
| 前端 | Python | Rust | L2 | 类 |
|---|---|---|---|---|
| models.routing.get | rpc_models.py:258 | 无(故意不桥接,设计允许) | 无 | 已知缺口 |
| models.routing.set | :268 | 最近似 config.routing.set_rule(已注册 config.rs:564) | 无 | C(需适配) |

### usage.*
| 前端 | Python | Rust | L2 | 类 |
|---|---|---|---|---|
| usage.status | rpc_usage.py:612 | 已注册(http_api.rs:789,形状不同) | 无 | A |
| usage.query | :727 | 无(summary/list/by_model/by_session 可聚合) | 无 | C(需聚合层) |
| usage.cost | :747 | 已注册(http_api.rs:808) | 无 | A |

### meta.*
| 前端 | Python | Rust | L2 | 类 |
|---|---|---|---|---|
| meta.runs.confirm_preflight | rpc_meta_runs.py:234 | 已实现未注册(meta_runs.rs:141) | 无 | B |
| meta.runs.replay | :283 | 已实现未注册 | 无 | B |
| meta.runs.list/show/failures/draft/diff/cost/validate/eval_baseline | :174-316 | 已实现未注册 | 无 | B |

## P6. 特殊发现

1. **两层缺口是普适的,不止 onboarding**:`crates/gateway/src/` 下 20 个 `register_*_handlers`,11 个从未被 `Gateway::new` 调用;`src-tauri` 41 个命令只覆盖网关面约 30%。
2. **sandbox 方法面完全正交**:Python(用户配置面)与 Rust(内部策略面)零重叠,无法桥接只能移植。
3. **plans.\* ≠ proposals.\***:`plans.*` 内嵌在 `rpc_sessions.py:7189-7696`(会话级计划模式);`exec.proposals.*`(rpc_proposals.py)是另一套 skill 提案系统。Rust 的 `proposals.rs` 死代码对应后者,**不解决 plans***。
4. **`sessions.subscribe` 语义差异**:Rust 是 long-poll,Python 是 WebSocket 订阅注册——即使绑定,事件推送模型也不同。
5. **`channels.pairing.revoke` 行为差异**:Python 版同步移除 admin 回调,Rust 只置 `enabled=false`——需按 Python 语义补齐。
6. **`usage.*` Rust 反而比 Python 多**(record/summary/by_model/by_session 等 Python 没有),前端要的 `usage.query` 是 Python 独有的时间窗口聚合,需新增。

## P7. 修复优先级建议(全应用视角)

按"影响面 × 修复成本"排序:

1. **P0 结构性收口**:把 11 个死代码 handler 模块的 `register_*_handlers` 挂进 `Gateway::new`(与 onboarding 一起做);同时给 `channels.*/cron.*/usage.*/sessions.subscribe/reset` 补 L2 绑定(类 A,成本最低,收益覆盖整个页面)。
2. **P0 会话列表**:`sessions.subscribe`(影响全站会话列表自动刷新)。
3. **P1 页面级可用**:`workspaces.list`(project 功能不可见)、`cron.*`(定时任务全不可用)、`plans.*`(Plan 模式全不可见,纯前端守卫已隐藏)、`agents.*`(Agents 增删改)。
4. **P2 移植类 C**:`sandbox` 用户面(正交,量大)、`channels.admin.set/restart/probe`、`sessions.steer.v2`、`models.routing.set`(可复用 config.routing.set_rule)、`usage.query`(可聚合 usage.summary 等)。
5. **P3 命名对齐(类 D)**:cron 别名、workspaces.remove→delete、migration.sources.*→discover/preview——可在 L2 适配层直接映射,不动网关。
6. **方法论**:与 onboarding 同——每个方法面先列"四维契约"(方法名/参数形状/返回形状/事件通道),再按"L1 网关接线 → L2 Tauri+注册表绑定 → 前端适配"三明治打通;静态复核 + gh CLI 触发 CI 验证。
