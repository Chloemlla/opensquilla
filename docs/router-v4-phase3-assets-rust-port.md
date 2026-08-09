# SquillaRouter V4 Phase 3 二进制资产解析与 Rust 复刻方案

> 针对 `src/opensquilla/squilla_router/models/v4.2_phase3_inference` 下
> `lgbm_model.bin / joblib / pkl` 等训练产物的作用详解，以及在 Rust 中完整
> 复刻其功能的方案。

## 1. 概述

`v4.2_phase3_inference` 是一个**本地"模型路由器"**（SquillaRouter V4 Phase 3）。
对每一轮用户消息：

1. 抽取一个 **390 维特征向量**；
2. 跑三个"头"（主 LightGBM、辅助 LightGBM、MLP）；
3. 按 per-class alpha 融合概率；
4. 经过 6 层规则后处理；
5. 归入 **R0–R3** 四个难度档，映射成模型 tier（S/M/L/XL）、思考模式
   （T0–T3）、提示词策略（P0–P2）和最终选择的模型。

即"这条消息该发给哪个模型/用多强推理"的自动决策器。

核心源码参考：

- 在线推理：`runtime_src/src/router/inference/`（`core.py` 编排、
  `features.py` 组装 390 维、`heads.py` 三头、`ensemble.py` 融合、
  `postprocess.py` 后处理）
- 特征提取：`runtime_src/src/router/features.py`（v3 51 维 HC + TF-IDF）、
  `runtime_src/src/router/v4_features.py`（BGE×3 PCA、assistant HC、
  continuation、reasoning 通道）
- 控制器：`src/opensquilla/squilla_router/v4_phase3.py`

## 2. 资产清单与作用

| 资产 | 大小 | 格式/对象 | 作用 |
|---|---|---|---|
| `lgbm_main.bin` | 39.7MB | LightGBM GBDT 二进制 | **主头**：输入 390 维特征 → R0–R3 四类概率。核心分类器 |
| `lgbm_aux.bin` | 3.5MB | LightGBM GBDT 二进制 | **辅头**：4 类 `initial/maintain/upgrade/downgrade`，供 aux 降档门控。默认 `aux_head_inference: false` 关闭 |
| `mlp/model.onnx` | 2.4MB | ONNX MLP | **第三头**：吃**原始 1536 维 BGE**（3×512，经 StandardScaler）→ 4 类 logits → `softmax(logits/temperature)`，`temperature=0.8092` |
| `mlp/scaler.joblib` | 37KB | sklearn `StandardScaler` | 1536 维 `mean_`/`scale_`，MLP 输入归一化 |
| `features/tfidf.pkl` | 344KB | sklearn `TfidfVectorizer`（char_wb、ngram 2–4、max_features=10000、sublinear_tf） | 文本 → 稀疏 TF-IDF |
| `features/svd.pkl` | 8MB | sklearn `TruncatedSVD`（≤100 主成分，random_state=42） | TF-IDF → ~100 维。`transform = X @ components_.T`，**不中心化** |
| `features/bge_pca.joblib` | 135KB | dict 包装内含 `PCA(64)`（来自 `BGEChannelExtractor.save()`） | 512 维 BGE → 64 维，作用在 3 个文本段 → 192 维 |
| `features/config.pkl` | 26B | `{"use_bge": ...}` | 旧提取器配置 |
| `features/meta.json` | — | schema 描述 | **注意：是旧的 151 维 schema（`use_bge:false`），勿信；以 `inference_manifest.json` 为准** |
| `bge_onnx/*` | 23.9MB(model) | BGE-small-zh-v1.5 ONNX + HF tokenizer | 512 维 CLS pooling + L2 归一化嵌入。同时喂 PCA 通道（进 GBDT）和原始 1536 维（进 MLP） |
| `inference_manifest.json` | — | 推理元数据 | `feature_dim=390`、`mlp_input_dim=1536`、`temperature=0.8092`、`per_class_alpha=[0.5,0.05,0.5,0.85]` |
| `router.runtime.yaml` | — | 全部规则配置 | 阈值、flag 关键词表、思考/提示/档位映射、aux/sticky 开关 |

## 3. 完整推理流水线

### 3.1 特征组装（390 维）

`inference/features.py::build_feature_bundle`：

```
hc(51) + tfidf(102, 补零) + ctx(10) + hist(16) + bge×3 PCA(192)
       + asst_hc(12) + cont(2) + reasoning(5) = 390
```

同时算 `raw_bge_1536 = 3×512`（BGE 原始嵌入，未 PCA）供 MLP 头。

各通道来源：

- **HC(51)**：`features.py::extract_handcrafted` — 长度/字数/行数、中英码
  比例、代码块/JSON/YAML/CSV/表格、标点、9 组关键词计数、文件/URL/日志/
  shell/traceback 正则、引号比例、词类覆盖、R1 教学/实现意图、文件引用
  分桶、长度分桶、低关键词密度。
- **TFIDF(102)**：`TfidfVectorizer.transform` → `TruncatedSVD.transform`，
  零填充到 102。
- **CTX(10)**：`extract_context_features` — turn_index/context_tokens/n_tools/
  tool_result_length 归一化 + 4 个布尔 + 2 个派生信号。
- **HIST(16)**：`extract_hist_features` — prev_route/difficulty/margin、
  max_route、turn_index、history_len、dominant_route、switches + 8 类
  trajectory one-hot。
- **BGE×3 PCA(192)**：`v4_features.py::BGEChannelExtractor.transform_one` —
  对 current_user、history_user（SEP 拼接最近 4 轮）、prev_assistant 三个
  文本段分别编码，各过 `PCA(64)`。
- **ASST_HC(12)**：`extract_assistant_handcrafted` — 澄清/拒答/自我怀疑/
  代码块/步骤列表正则 + usage 归一化（log1p/10）+ 长度比、中文字符比、
  缓存 token 比。
- **CONT(2)**：`extract_continuation_features` — 短续写提示 cue + 前轮输出
  token 对数。
- **REASONING(5)**：`extract_reasoning_features` — 推理 cue + 问号密度 +
  长度对数 + 前轮推理 token/耗时对数。

### 3.2 三头推理（`heads.py::run_heads`）

```
p_main = lgbm_main.predict(features_390)          # 4 类概率
p_aux  = lgbm_aux.predict(features_390)           # 4 类（initial/maintain/upgrade/downgrade）
logits = mlp(scaler(raw_bge_1536))                # 4 类 logits
p_mlp  = softmax(logits / temperature)            # temperature = 0.8092
```

### 3.3 融合（`ensemble.py::fuse_probabilities`）

```
fused[i] = alpha[i]*p_main[i] + (1-alpha[i])*p_mlp[i], 再归一化
alpha = [0.5, 0.05, 0.5, 0.85]
```

注意 **R1 档几乎全由 MLP 决定**（alpha 仅 0.05）。

### 3.4 后处理（`postprocess.py::apply_postprocess`）

argmax → margin 升级 → aux 降档（可选）→ R1 救援 → 防欠路由安全网 →
flag 覆盖 → 上下文规则 → sticky 粘档（可选）。

关键阈值（`router.runtime.yaml`）：

- `margin_upgrade: 0.10`
- `r1_rescue.from_r0_max_gap: 0.10`
- `under_routing_safety: 0.45`
- `v4.aux_downgrade`: `enabled: false`, `threshold: 0.55`
- `v4.sticky_tier`: `enabled: false`, `max_user_len: 200`

派生（`predictor.py` / `postprocess.py`）：

- 思考模式 T0–T3、提示策略 P0–P2（含 trivial-ack 特判：R0 且是
  thanks/收到/好的 等 → 强制 T0/P0）
- tier = `tier_mapping[route_class]`，模型 = `tier_registry[tier][0]`

flag（`flags.py::compute_flags`）：high_risk / debug / repo_arch /
strict_format / long_context，全部由 `router.runtime.yaml` 的 `flag_rules`
关键词表和阈值驱动。

## 4. Rust 复刻方案

核心认识：**这些是训练产物，Rust 里不用重新训练，只需加载拟合参数 + 重写
固定的推理算法**。算法全是确定性的。难点只在序列化格式。

按格式分四类处理：

| 资产 | 格式 | Rust 处理方式 |
|---|---|---|
| BGE、MLP `.onnx` | 可移植 ONNX | **复用现有 `ort`**。`squilla_inference.rs::embed` 已完成 BGE CLS+归一化；MLP 头另开一个 session |
| `lgbm_*.bin` | LightGBM 自有二进制（**不是 pickle**） | 首选 Rust `lightgbm` crate（FFI 绑定 lib_lightgbm）直接加载 + predict，与 Python 位级一致；备选一次性导出 ONNX 走 `ort` |
| `*.pkl` / `*.joblib` | Python pickle（= 任意代码执行） | **Rust 不读 pickle**。一次性转换脚本把拟合参数 dump 成纯数据，再手写矩阵乘（矩阵都很小） |
| 特征算法 + 规则后处理 | 纯逻辑 | 全部手写 |

### 4.1 转换脚本（一次性，Python）

`scripts/export_router_params.py`：读 .pkl/.joblib，输出 `params/` 下的
JSON/二进制，并合并更新 `inference_manifest.json`（Rust `Manifest` 读取的
`temperature`/`per_class_alpha`；`artifact_manifest.json` 是校验和/provenance
清单，由 `update_router_artifact_manifest.py` 维护，导出脚本不覆盖）。需导出
的参数：

| 源资产 | 导出内容 | 备注 |
|---|---|---|
| `features/tfidf.pkl` | `vocabulary_`（词→id）、`idf_` 数组 + 构造参数（analyzer=char_wb、ngram_range=(2,4)、sublinear_tf） | Rust 重写 char_wb 分词 + idf 乘 + L2 行归一化 + sublinear tf |
| `features/svd.pkl` | `components_`（~100 × 10000 稀疏矩阵） | `transform = X @ components_.T`，**不中心化** |
| `features/bge_pca.joblib` | 解包 dict：`pca.components_`（64×512）、`pca.mean_`（512） | `transform = (X - mean_) @ components_.T`，**要中心化** |
| `mlp/scaler.joblib` | `mean_`、`scale_`（各 1536） | `transform = (X - mean_) / scale_` |
| `features/config.pkl` | 直接转 JSON | 26 字节 dict |
| `lgbm_*.bin` | 原样保留（`lightgbm` crate 直接读）或导 ONNX | — |

### 4.2 Rust 侧新模块建议

```
crates/engine/src/squilla_router/
  features/     # handcrafted(51)、asst_hc(12)、cont(2)、reasoning(5)、
                # ctx(10)、hist(16)+trajectory、char_wb TF-IDF、SVD/PCA/scaler
  heads/        # LightGBM main/aux + ONNX MLP + temperature softmax
  ensemble.rs   # alpha 融合（已存在 fuse_probabilities）
  postprocess.rs# 6 层后处理 + thinking/prompt/tier 派生 + compute_flags
  config.rs     # 解析 router.runtime.yaml
```

## 5. 当前 Rust 差距（`crates/engine/src/squilla_inference.rs`）

- ✅ 已移植：BGE `embed`（`ort`）、`fuse_probabilities`、
  `apply_post_processing`（部分）
- ❌ 缺口：
  - `route()` 是 stub，直接返回 `HeadsMissing`
  - 三个头（lgbm_main / lgbm_aux / MLP）都没接
  - 全部特征提取缺失（390 维 + raw 1536）
  - `RoutingFlags` 只有 5 个 bool，没有 yaml 关键词表
  - `apply_post_processing` 阈值写死（0.15/0.20/0.45），与 yaml
    （0.10/0.10/0.45）不一致
  - aux 降档、上下文规则、sticky 粘档、trivial-ack、thinking/prompt/
    tier 派生缺失

## 6. 建议落地路径

1. **转换脚本**：导出 TF-IDF/SVD/PCA/scaler 参数 + 更新 manifest。
2. **特征前端**：移植全部 HC 逻辑 + char_wb TF-IDF + SVD/PCA/scaler 矩阵乘。
3. **三个头**：`lightgbm` crate 加载 main/aux；`ort` 加载 MLP + temperature
   softmax；按 manifest 的 alpha 融合。
4. **后处理补齐**：改成从 `router.runtime.yaml` 读配置，补齐 aux/sticky/
   compute_flags/派生逻辑。

## 7. 注意

- `features/meta.json` 描述的是旧 151 维 schema（`use_bge:false`、
  `channel_order: [HC, TFIDF]`），与 390 维在线组装无关，**勿信**。
- `.pkl`/`.joblib` 是 pickle，等于任意代码执行（`PROVENANCE.md` 已注明）。
  Rust 迁移天然不碰 pickle，这是安全收益。
- 训练侧在 `src/opensquilla/squilla_router/self_learning/train.py`
  （`lgb.train` + sklearn fit + MLP）。建议训练仍留在 Python（离线运维操作），
  Rust 只做推理。
