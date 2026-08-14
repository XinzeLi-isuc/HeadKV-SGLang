# HeadKV-SGLang S 级升级计划书

> 前置文档:`HeadKV-SGLang_修订版项目计划书.md`(MVP 版,已完成,A 级达成)。
> 本文档定义从 A 级升级到 S 级的目标、方案、阶段、Gate、实验与工作量。
> 升级基线:MVP commit(fork `feat/headkv-duo` @ 03a66988;项目仓库 @ f3c8f11)。

## 1. 升级背景

### 1.1 MVP 完成状态(A 级,2026-08-13 全部 Gate 通过)

```text
容量:    T0=204824 → Tf=782432(ratio 0.25, 3.82x)/ 397360(0.5, 1.88x)
        理论-实测 0.0000% 偏差;comp 池仅占预算 3%
正确性:  56 单测全绿;20/20 逐 token 一致(短 prompt)
质量:    NIAH 4K 9/9 无损;LongBench 2 子任务 F1 持平
生命周期:1500 混合请求零失败,两轮吞吐无衰减
实验:    Exp A-E 全执行;单请求开销 = 0.33s 固定启动成本(不随长度增长)
```

### 1.2 升级动机

1. **面试区分度**:A 级叙事是"接入/集成";S 级"policy-decoupled 后端"是
   独立架构贡献(HeadPolicy 抽象 + 两种 head 选择算法的同一后端对照)
2. **研究型卖点**:DuoAttention(score 阈值)与 RLKV(adapter 学习)是两种
   完全不同的 head 选择算法,在**同一 runtime** 下的分布差异与收益对比
   是现成的研究素材,可量化、可作图
3. **吞吐短板补齐**:分析报告定位 0.33s 固定开销 → CUDA Graph 开启可直接
   消除并量化,补齐 A 级唯一的性能短板
4. **计划书 §10 定义**:满足"同一 runtime 双 policy"或"current-main 迁移"
   其一即达 S 级;路线 1 代码 80% 已就绪,是短平快路径

## 2. S 级目标与判定

### 2.1 目标

```text
HeadKV-SGLang:Policy-decoupled Head-wise Heterogeneous KV Backend
```

在 A 级基础上,把 RLKV 老入口(`--enable-rlkv-inference`)也收敛到
HeadPolicy 工厂,使 runtime 完全 policy 无关;用同一套双池后端跑通
DuoAttentionPolicy 与 RLKVPolicy 的完整对照;并开启 CUDA Graph 消除
固定开销。

### 2.2 S 级判定(Gate 全过后)

- [x] 同一 runtime 同时支持 DuoAttentionPolicy 与 RLKVPolicy
      (HeadPolicy.create 工厂统一分发,无 runtime 内 policy 分支)
- [x] 双 policy 容量/质量/吞吐对照实验完成(数据表 + 图)
- [x] RLKVPolicy 与原 RLKV loader 输出一致(去随机后,确定性)
- [x] head 分布差异分析(Duo vs RLKV 的 full head 分布对比)
- [x] (增强)CUDA Graph 开启,固定开销消除量化
- [ ] (可选)current SGLang main 端到端迁移(两天止损)

## 3. 技术方案总览

```text
Phase S0  双 policy 接入(runtime 统一入口)
Phase S1  双 policy 对照实验(A 容量 + E 质量 + B/C 吞吐)
Phase S2  head 分布差异分析(研究型卖点)
Phase S3  CUDA Graph 开启 + 性能复测
Phase S4  (可选)current-main 迁移(两天止损)
Phase S5  文档、叙事、交付
```

推荐顺序:S0→S1→S2→S3 为 S 级必做;S4 为投递后深化(独立 2~4 天)。

---

## 4. Phase S0:双 policy 接入(0.5 天)

### 4.1 现状盘点

```text
已有(未接入):
  python/sglang/srt/headkv/rlkv_policy.py —— RLKVPolicy 完整实现
    (load adapter_weights.tsv / full_attention_heads.tsv、clip、
     sparsity-quantile 二值化、确定性 tie-break,无随机微扰)
仍直连(老入口):
  model_runner._load_rlkv_head_masks() —— enable_rlkv_inference 分支
    (L1552-1597,含 np.random.uniform 微扰,不可复现)
```

### 4.2 改动点

| 文件 | 改动 |
| --- | --- |
| `server_args.py` | `enable_rlkv_inference` 保持兼容,内部转 HeadKVConfig(policy="rlkv");互斥校验改为二者均可,但 policy 字段唯一 |
| `model_runner.py` | `_init_headkv()` 统一处理两个入口:cfg.policy 由 server_args 推导(enable_headkv→duo;enable_rlkv_inference→rlkv);`_load_rlkv_head_masks` 标记废弃(保留兼容,日志提示) |
| `headkv/config.py` | HeadKVConfig 增加 `sparsity` 字段(RLKVPolicy 参数);validate 分支 |
| `test/` | 新增 `test_rlkv_policy.py`(确定性/sparsity 语义/shape 校验/与 duo 的 mask 差异) |

### 4.3 兼容性要求

- `--enable-rlkv-inference` 单独使用:行为与原版一致(除了随机微扰 → 确定性,
  输出 mask 可能微变;需一致性测试记录差异)
- `--enable-headkv --headkv-policy rlkv --headkv-rlkv-sparsity 0.5`:
  新入口显式选 RLKV
- 两者同时开启:报错(policy 冲突)

### 4.4 Gate S0

- [x] RLKVPolicy 单测全过(确定性:两次运行 mask 相同;sparsity 语义:
      threshold = quantile(scores, sparsity);shape 校验)
- [x] 老入口 RLKV server 启动 + 单请求生成正常
- [x] 新入口 RLKV server 启动 + 单请求生成正常
- [x] 两个入口的 mask 一致性记录(mask 覆盖率、full head 集合 Jaccard;
      差异仅来自随机微扰)

---

## 5. Phase S1:双 policy 对照实验(0.5~1 天)

### 5.1 实验设计(全部沿用 MVP 协议)

| 实验 | Duo(ratio 0.5) | RLKV(sparsity 0.5) | 输出 |
| --- | --- | --- | --- |
| A 容量 | Tf 实测 | Tf 实测 | capacity 对照表 |
| E 质量 | NIAH 4K + LongBench 2 子任务 | 同左 | 质量对照表 |
| B 单请求 | 4K/8K/16K E2E | 同左 | 开销对照 |
| C 并发 | 8K×bs sweep | 同左 | 并发对照 |

### 5.2 关键预期(假设,实验后核实)

- 容量:RLKV 的 sparsity-quantile 保证 **恰好 50% full**,而 Duo 的
  threshold=0.5 是 91.4% full(Phase 2 实测)—— 两者 effective ratio 不同,
  容量不同;对照应标注 effective ratio 而非名义值
- 质量:RLKV 的 50% full 可能比 Duo(91% full)信息保留少,但 head 选择更
  精准(学习型)→ 同 effective ratio 下的质量对比是真正的算法对比
- **建议补一组 Duo(threshold=0.5, 91% full)vs RLKV(sparsity 0.09)的
  同 effective ratio 对照**,消除容量差异干扰,直比 head 选择质量

### 5.3 Gate S1

- [x] 双 policy 容量/质量/吞吐对照表(artifacts/s_policy_compare.csv)
- [x] 同 effective ratio 质量对照(可选但推荐)
- [x] 无崩溃/无泄漏(复用 Phase 6 协议,每 policy 300 请求)

---

## 6. Phase S2:head 分布差异分析(0.5 天)

### 6.1 分析内容

1. Duo 与 RLKV 的 full head 集合对比:逐层 full 数、层间差异、Jaccard
2. 可视化:heatmap(层 × head 的 full/compact 分布)双图并排
3. 洞察:两种算法的 head 选择是否收敛到相似子集(高 Jaccard = 算法共识,
   低 Jaccard = 互补信息 → 未来可做 head 级集成)

### 6.2 交付

- figures/s_head_distribution_duo.png / _rlkv.png / _overlap.png
- docs/headkv/policy_analysis.md(分布统计 + 洞察)

### 6.3 Gate S2

- [x] 分布图 + 统计表(Jaccard、逐层 full 数)
- [x] 洞察段落(面试叙事素材)

---

## 7. Phase S3:CUDA Graph 开启(0.5~1 天)

### 7.1 背景

MVP 用 eager(MVP 明确不做 CUDA Graph)。分析报告定位单请求开销 =
0.33s 固定启动成本(双路 kernel + gather/restore + metadata)。CUDA Graph
可消除 kernel launch 与部分 metadata 开销。

### 7.2 方案

1. 移除 `--disable-cuda-graph`,开启默认 CUDA Graph(backend 已有
   `init_cuda_graph_state/capture/replay` 实现,见 head_realloc_backend
   L538-650;MVP 未启用)
2. 按计划书 Phase 6 要求:小规模重复测试(4K/8K 单请求 + 20 条 prompts
   正确性回归)
3. 性能复测:Exp B 协议(4K/8K/16K E2E + decode TPOT)+ Exp C 并发

### 7.3 风险与回退

- CUDA Graph 与 HeadKV 双池的 window metadata 交互:replay 路径依赖
  `_update_comp_mapping_decode`(backend 已实现)
- 若正确性/性能不达标:回退 eager,记录原因,不影响 S 级判定
  (S 级判定不依赖 CUDA Graph)

### 7.4 Gate S3

- [x] CUDA Graph 下 20/20 正确性回归(与 eager FullKV 首 token 一致率 ≥95%)
- [x] 单请求 E2E 复测:0.33s 固定开销的消除量化(目标:绝对差显著下降)
- [x] 并发吞吐复测(期望:memory-light 场景 0.78x → 接近 1.0x)
- [x] 300 请求生命周期回归

---

## 8. Phase S4:(可选)current-main 迁移(2~4 天,止损)

沿用原计划书 §7 框架,基于 MVP 经验更新:

### 8.1 接入点(更新)

```text
1. ServerArgs/config validation   —— HeadKVConfig 适配 current ServerArgs
2. memory pool registry/factory   —— current main 的 KVCache 结构变化
3. allocator component            —— headkv/budget + allocator 语义
4. attention backend setup factory —— current main backend 注册方式
```

### 8.2 可复用资产(MVP 已验证)

- headkv/ 包(纯算法,零 SGLang 依赖)—— 直接复用
- 预算公式与测试(56 单测)
- 实验协议与脚本

### 8.3 止损

两天无法打通端到端 decode → 停止,保留 v0.5.2 版本,不影响 S 级
(双 policy 已满足 S 级判定)。

---

## 9. Phase S5:文档与交付(0.5 天)

- README 更新:S 级状态、双 policy 对照表、CUDA Graph 结果
- docs/reports/phase-s0..s5.md 每阶段报告
- 面试叙事更新:narrative.md 增加双 policy 故事 + head 分布洞察
- git:fork 分支 + 项目仓库逐 commit

---

## 10. 工作量账本

| 任务 | 人日 | 方差 | 前置 | 产出 |
| --- | ---: | --- | --- | --- |
| S0 双 policy 接入 | 0.5 | ±0.25 | - | 统一入口 + RLKVPolicy 单测 |
| S1 双 policy 对照实验 | 0.5~1 | ±0.5 | S0 | 对照表 + 同 ratio 质量对比 |
| S2 head 分布分析 | 0.5 | ±0.25 | S1 | 3 张图 + 分析文档 |
| S3 CUDA Graph | 0.5~1 | ±0.5 | S1 | 开销消除量化 + 回归 |
| S4 current-main 迁移 | 2~4 | ±1(止损) | 可选 | 两天 Gate |
| S5 文档交付 | 0.5 | ±0.25 | S0-S3 | README/报告/叙事 |

**S 级必做(S0-S3+S5):2.5~3.5 人日,方差 ±1.5,对照 5 天日历可行**
**含 S4:4.5~7.5 人日**

## 11. 执行日历(5 天)

| 日期 | 主任务 | 当天必须交付 |
| --- | --- | --- |
| Day 1 | S0 双 policy 接入 | RLKVPolicy 单测 + 双入口 smoke |
| Day 2 | S1 对照实验(A/E) | 容量与质量对照表 |
| Day 3 | S1 剩余(B/C)+ S2 分布分析 | 吞吐对照 + 分布图 |
| Day 4 | S3 CUDA Graph | 0.33s 消除量化 + 回归 |
| Day 5 | S5 文档交付 | S 级证据链完整 |

## 12. 风险与止损矩阵

| 风险 | 概率 | 影响 | 止损 |
| --- | --- | --- | --- |
| RLKVPolicy 接入破坏老入口 | 中 | 低 | 老入口保留原代码路径,新入口并行,对照一致后再切换 |
| 同 effective ratio 对照无显著差异 | 中 | 低 | 如实报告(两种算法质量相当也是结论);head 分布分析补叙事 |
| CUDA Graph 与双池不兼容 | 中 | 中 | 回退 eager,记录;S 级判定不依赖它 |
| S1 实验时间超支 | 中 | 低 | 砍 B/C 到最小集(4K/8K 两点) |
| current-main 迁移失败 | 中 | 高(仅 S4) | 两天 Gate 止损,双 policy 已满足 S 级 |

## 13. 最终叙事(升级版)

```text
DuoAttention 与 RLKV 用完全不同的方法识别关键 KV heads——
前者基于 attention score 阈值,后者基于训练 adapter。
两者部署时都归约为 Full/Compact 两种生命周期。
本项目把 RLKV 算法专属的 mask loader 与 runtime 解耦,
抽象出 policy-decoupled 的 HeadKV 后端:
同一套双池 runtime 可以装载任意 head 选择策略,
并在同一后端下完成两种算法的容量/质量/分布对照,
验证了 head-wise 异构 KV 缓存作为通用 serving 原语的可行性。
```

面试亮点:
- 0.33s 固定开销定位 → CUDA Graph 消除(完整的问题-归因-解决闭环)
- 双 policy 同一 runtime:架构贡献而非算法搬运
- head 分布 Jaccard:两种算法的共识/互补,可现场展开

## 14. 参考

- MVP 计划书 §7/§10/§11(升级框架与叙事)
- docs/reports/analysis.md(0.33s 固定开销归因)
- docs/headkv/known_limitations.md(K1-K9 升级候选)
