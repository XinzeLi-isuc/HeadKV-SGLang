# HeadKV-SGLang 修订版项目计划书

> 项目第一阶段的准确名称建议使用 **DuoKV-SGLang**。只有在 DuoAttention 与 RLKV 两种 policy 都能复用同一套 runtime，或完成 current SGLang 的通用化迁移后，再升级为 **HeadKV-SGLang**。

## 1. 修订结论

原计划的研究方向成立，但执行路线过于激进。最关键的问题不是 DuoAttention policy，而是把 RLKV 基于 SGLang v0.5.2 的 head-wise runtime 迁移到 current SGLang。current main 的 memory pool、allocator、attention backend 构建流程已经发生较大变化，不能把现有 Full/SWA 基础设施直接当成“现成的 head-wise 双池”。

因此，项目改成两级路线：

1. **主线 MVP：基于 RLKV 已验证的 SGLang v0.5.2 fork，抽离 RLKV 专属 head loader，接入 DuoAttention policy，完成真实物理双池、连续批处理、容量与质量验证。**
2. **升级路线：MVP 通过后，再把算法无关的 HeadPolicy、预算模型和双池接口迁移到冻结的 current SGLang。**

这样可以先得到一个真实可运行、可写进简历的系统项目，再决定是否承担 current-main port 的架构风险。

---

## 2. 原设计中需要删除或修改的内容

| 原设计 | 问题 | 修订 |
| --- | --- | --- |
| 五天内完成 current main 双池、continuous batching 和质量实验 | current main 与 RLKV v0.5.2 的组件边界差异较大，估时不成立 | 先用 RLKV fork 完成 7～8 天 MVP；current-main port 单列升级阶段 |
| 优先直接泛化 current SGLang Full/SWA allocator | Full/SWA 的异构单位主要是 layer，而 HeadKV 是同一 layer 内的 KV head；只能借鉴协议，不能直接复用存储布局 | MVP 复用 RLKV 的 `HeadReallocKVPool`、`HeadReallocAllocator` 和 backend；current main 只借鉴 rollback、location translation、pool registry 设计 |
| DuoAttention 的 Q-head mask 先做 GQA OR 聚合 | 官方 DuoAttention pattern 对 Llama-3 等模型已经是 `[num_layers, num_kv_heads]`；再次 OR 会改变 pattern 并降低压缩率 | 首先严格校验 pattern 维度。只有外部输入确实是 Q-head mask 时才执行 OR 转换 |
| 使用 RLKV 默认 sink/recent window | DuoAttention pattern 目录已有自己的 `config.json`，其 window 可能与 RLKV 默认值完全不同 | 默认读取 DuoAttention `config.json`；命令行覆盖必须显式记录 |
| RLKV loader 的随机微扰打破相同分数 | 会造成 mask 不可复现 | 使用稳定排序，以 `(score, layer_id, head_id)` 确定固定 top-k |
| Prefix Cache 是主目标组成部分 | RLKV fork 当前默认关闭 Radix Cache；Head-wise compact state 的 prefix 复用是独立项目 | MVP 明确关闭 Prefix Cache，不影响项目成立 |
| 一开始同时验证 CUDA Graph | 增加 metadata 和 replay 调试变量 | eager correctness 与 allocator lifecycle 通过后再打开 CUDA Graph |
| Official DuoAttention、HF 和 SGLang 都作为性能 baseline | 三者 runtime 不同，吞吐数据不可直接归因 | Official DuoAttention只做算法正确性/质量 oracle；系统性能只比较同一 SGLang commit 的 FullKV 与 DuoKV |
| 自动推导 `max_num_reqs` | RLKV compact pool 按 `max_num_reqs × window` 分配；自动值可能达到数千，导致 compact pool 过大甚至 OOM | 所有实验显式设置 `--max-running-requests`，并把它写进日志 |

---

## 3. 项目目标与边界

### 3.1 一句话目标

将 DuoAttention 的 Retrieval/Streaming KV-head pattern 接入 RLKV 的 SGLang head-reallocation runtime，使 Retrieval Heads 保存完整历史，Streaming Heads 仅保存 sink + recent，并将节省的显存重新分配给 Full Pool，从而提升 `max_total_tokens` 和连续批处理并发容量。

### 3.2 MVP 必须完成

- 单 GPU、TP=1；
- Llama/Mistral 类 decoder-only MHA 或 GQA 模型；
- DuoAttention 官方 pattern 加载、确定性二值化和维度校验；
- KV-head 级 Full/Compact partition；
- 复用 RLKV 的真实 `HeadReallocKVPool`；
- Compact Pool 只保留 sink + recent，不保存完整历史副本；
- eager decode、extend 和 continuous batching；
- allocator 释放、环形覆盖和 request interleave 正确；
- FullKV 与 DuoKV 的真实 token capacity 对比；
- NIAH 和至少两个 LongBench 子任务；
- 同一 SGLang runtime 下的 throughput/concurrency 实验。

### 3.3 MVP 明确不做

- current SGLang main 的端到端迁移；
- Prefix/Radix Cache；
- speculative decoding；
- TP/PP/DP；
- PD disaggregation、KV offload；
- 动态 head classification 或 DuoAttention pattern 训练；
- RLKV 的 GRPO 训练；
- FP8/INT4 KV；
- 自定义融合 kernel；
- upstream PR。

这些能力只有在 MVP 的物理容量收益和正确性已经成立后才能进入升级阶段。

---

## 4. 冻结的技术路线

### 4.1 主基线

- Runtime：RLKV 官方 SGLang fork，基于 SGLang v0.5.2；
- Attention：`HeadReallocAttnBackend`；
- KV storage：`HeadReallocKVPool`；
- Allocator：`HeadReallocAllocator`；
- Policy：新增 `DuoAttentionPolicy`；
- FullKV baseline：相同 commit、相同模型、相同 dtype、相同 backend 约束；
- 算法 oracle：官方 DuoAttention 实现及其官方 pattern。

项目开始时必须执行并记录：

```bash
git rev-parse HEAD
python -V
python -c "import torch; print(torch.__version__, torch.version.cuda)"
nvidia-smi
```

禁止在实验过程中继续 `git pull`。

### 4.2 首选模型

主模型建议：

```text
Meta-Llama-3.1-8B-Instruct
```

原因：DuoAttention 官方提供 pattern，且能验证 GQA。单卡 A6000 48GB 先做 8K/16K/32K；实际最大长度以显存和模型许可为准。

如果希望更明显地展示 head-wise 压缩上限，可增加一个 MHA 次模型：

```text
Llama-2-7B-32K-Instruct
```

但次模型不阻塞主项目。

### 4.3 固定运行约束

第一轮统一：

```text
TP = 1
page_size = 1
Prefix Cache = OFF
CUDA Graph = OFF
Speculative Decoding = OFF
temperature = 0（正确性）
dtype = FP16/BF16，FullKV 与 DuoKV 保持一致
max_running_requests = 显式指定
```

---

## 5. 最终代码框架

在 RLKV fork 上保持小改动，不大范围重构 SGLang：

```text
python/sglang/srt/
├── headkv/
│   ├── __init__.py
│   ├── config.py          # HeadKVConfig 和参数合法性
│   ├── policy.py          # HeadPolicy 抽象接口
│   ├── duo_policy.py      # Duo pattern/config loader
│   ├── rlkv_policy.py     # 可选：兼容原 RLKV loader
│   ├── partition.py       # global KV head → TP-local KV head
│   └── budget.py          # Full/Compact pool 容量计算
│
├── layers/attention/
│   └── head_realloc_backend.py  # 尽量复用，仅修通用命名与边界 bug
│
├── mem_cache/
│   ├── memory_pool.py     # 复用 HeadReallocKVPool
│   └── allocator.py       # 复用 HeadReallocAllocator
│
├── model_executor/
│   └── model_runner.py    # 从 HeadPolicy 获取 mask 并初始化双池
│
└── server_args.py         # 新增通用 HeadKV 参数，保留 RLKV 兼容入口

test/srt/headkv/
├── test_duo_policy.py
├── test_partition.py
├── test_budget.py
├── test_headkv_pool.py
├── test_headkv_allocator.py
├── test_headkv_attention.py
└── test_headkv_lifecycle.py

docs/headkv/
├── env.md
├── architecture.md
├── rlkv_callgraph.md
├── experiment_protocol.md
└── known_limitations.md

benchmarks/headkv/
├── run_capacity.py
├── run_offline_throughput.py
├── run_online_serving.py
├── run_niah.py
└── configs/
```

### 5.1 核心接口

```python
class HeadPolicy:
    def load_global_kv_mask(self, model_config):
        """返回 bool mask，shape=[num_layers, global_num_kv_heads]。
        True 表示 Full，False 表示 Compact。
        """

    def sink_size(self) -> int: ...
    def recent_size(self) -> int: ...
```

Runtime 只消费：

```text
KV-head mask + sink size + recent size
```

它不关心 mask 是 DuoAttention、RLKV 还是人工配置产生的。

---

## 6. 分阶段执行计划

## Phase 0：环境冻结与 FullKV 基线

时间：0.5 天。

### 要做什么

1. 固定 RLKV SGLang fork commit；
2. 安装与官方 fork 匹配的 Torch/CUDA/Triton 依赖；
3. 下载主模型和对应 DuoAttention pattern；
4. 先跑 Vanilla SGLang FullKV；
5. 再跑官方 DuoAttention 单请求生成；
6. 固定一组 20 条 correctness prompts。

### 具体怎么做

- 使用单卡 A6000；
- FullKV 分别跑 4K、8K prompt smoke test；
- 记录启动参数、模型 revision、GPU、峰值显存、首 token 和生成结果；
- 显式设置 `max_running_requests=32` 作为最初 smoke 配置，避免自动值污染 compact pool 预算；
- 不进行吞吐调优。

### 产物

```text
docs/headkv/env.md
artifacts/fullkv_smoke.log
artifacts/official_duo_smoke.log
benchmarks/headkv/prompts_correctness.jsonl
```

### Gate 0

必须同时满足：

- FullKV 能稳定生成；
- Official DuoAttention 能使用同一模型/pattern 生成；
- 记录的 pattern shape 与模型 `num_layers × num_kv_heads` 一致。

如果半天仍无法跑通，停止写 runtime，先锁定兼容的模型 revision 和依赖版本。

---

## Phase 1：逆向 RLKV serving 数据流

时间：0.5 天。

### 要做什么

只追踪以下调用链：

```text
mask loader
→ pool budget
→ HeadReallocKVPool
→ HeadReallocAllocator
→ HeadReallocAttnBackend
→ request free
```

### 具体怎么做

必须在文档中回答：

1. head mask 在哪个时刻加载和 TP shard；
2. Full/Compact tensor 的实际 shape；
3. `full_to_comp_mapping` 何时写入；
4. 每个 request 的 compact chunk 何时分配；
5. sink slot 和 recent ring slot 如何计算；
6. extend 如何构造 compressed prefix；
7. decode 如何覆盖 recent ring；
8. request finish/free-group 如何释放 compact chunk；
9. CUDA Graph 额外保存哪些 metadata；
10. compact pool 耗尽时当前代码如何处理。

### 产物

```text
docs/headkv/rlkv_callgraph.md
docs/headkv/architecture.md
docs/headkv/known_limitations.md
```

### Gate 1

能够从一个新 request 进入到释放，画出 Full loc、Compact loc 和 request slot 三者关系。不能解释完整生命周期时，不进入 policy 接入。

---

## Phase 2：实现算法无关的 HeadPolicy

时间：1 天。

### 要做什么

1. 定义 `HeadKVConfig`；
2. 定义 `HeadPolicy` 抽象；
3. 实现 `DuoAttentionPolicy`；
4. 将原 RLKV loader 封装成可选的 `RLKVPolicy`；
5. 实现确定性的 threshold/top-k 二值化；
6. 实现 TP-local partition 接口，但 MVP 只启用 TP=1。

### DuoAttentionPolicy 的具体规则

输入目录必须包含：

```text
full_attention_heads.tsv
config.json
```

加载流程：

```text
读取连续 head score
→ clip 到 [0,1]
→ 校验 [layers, global_kv_heads]
→ threshold 或指定 full-head-ratio 二值化
→ 稳定排序处理同分
→ 生成 bool KV-head mask
→ 读取 sink/recent
```

配置优先级：

```text
显式命令行覆盖
> deploy_sink_size/deploy_recent_size
> config.json 中 sink_size/recent_size
```

禁止静默使用 RLKV 默认 window。

### GQA 处理

- 官方 pattern 如果列数等于 `global_num_kv_heads`：直接使用；
- 只有列数等于 `global_num_q_heads` 时，才按共享 KV group 做 OR；
- 其他维度直接报错；
- backend 再将每个 KV head 展开为对应 Q-head group，不能打乱 Q/K/V 对应关系。

### 单元测试

```text
test_load_official_duo_pattern
test_pattern_shape_validation
test_deterministic_topk
test_window_config_priority
test_gqa_kv_pattern_no_second_aggregation
test_qhead_pattern_or_aggregation
test_tp1_partition
test_partition_complete_and_disjoint
```

### Gate 2

- 多次运行得到完全相同的 mask；
- Full + Compact 覆盖全部 KV heads 且无重叠；
- 日志打印 nominal ratio 和 effective KV-head ratio；
- 官方 GQA pattern 不被错误地再次 OR 聚合。

---

## Phase 3：把 DuoPolicy 接入 RLKV 双池 runtime

时间：1 天。

### 要做什么

新增通用参数：

```text
--enable-headkv
--headkv-policy duo|rlkv
--headkv-pattern-path PATH
--headkv-full-head-ratio R       # 与 threshold 二选一
--headkv-threshold T
--headkv-sink-size S             # 可选显式覆盖
--headkv-recent-size W           # 可选显式覆盖
```

原 `--enable-rlkv-inference` 保留为兼容入口，但内部也转换成 `HeadPolicy`，不能再让 `ModelRunner` 直接知道 RLKV adapter 文件格式。

### 接入调用流

```text
ServerArgs
→ HeadKVConfig.validate()
→ HeadPolicy.load_global_kv_mask()
→ TP-local mask
→ calculate_pool_budget()
→ HeadReallocKVPool
→ HeadReallocAllocator
→ HeadReallocAttnBackend
```

### 正确的预算公式

设：

- `T0`：FullKV profiling 得到的 baseline token capacity；
- `F`：所有层 Full KV-head 数之和；
- `C`：所有层 Compact KV-head 数之和；
- `R`：显式设置的 `max_running_requests`；
- `V = sink + recent`；
- `Tc = R × V`：Compact Pool token slots；
- `Tf`：重新分配后的 Full Pool token capacity。

保持 KV byte budget 近似不变：

```text
T0 × (F + C) = Tf × F + Tc × C
```

因此：

```text
Tf = floor((T0 × (F + C) - Tc × C) / F)
```

需要单独处理：

- `F=0`：不允许作为正式 serving 配置，或进入全 streaming 专用路径；
- `C=0`：退化为 FullKV，不分配 Compact Pool；
- `Tc × C >= T0 × (F+C)`：配置不可能，启动时报错；
- `R` 未显式设置：HeadKV 启动时报错，避免分配数千个 compact chunks。

### Gate 3

启动日志必须给出：

```text
policy / pattern path
layers / Q heads / KV heads
full and compact KV-head counts
sink / recent / window
max_running_requests
baseline T0
full pool Tf
compact pool Tc
predicted capacity gain
```

同时检查每层物理 tensor：

```text
Full:    [Tf + 1, F_layer, head_dim]
Compact: [Tc + 1, C_layer, head_dim]
```

不能存在额外的 `[T0, all_heads, head_dim]` 完整 KV 副本。

---

## Phase 4：Attention 语义正确性

时间：1～1.5 天。

### 4.1 Tensor-level reference

用纯 PyTorch 写最小 reference：

- Full heads：标准 causal attention；
- Compact heads：只允许访问前 `sink` tokens 和最近 `recent` tokens；
- MHA 与 GQA 分别测试；
- 序列长度覆盖 `< window`、`= window`、`> window`；
- 覆盖一次性 extend、chunked extend 和逐 token decode。

测试配置：

```text
all full
all compact
50% mixed
每层不同 mask
GQA shared KV groups
```

FP16 建议从以下容差开始：

```text
atol=1e-2, rtol=1e-2
```

如果超差，先定位数值路径，不能直接放宽到失去意义。

### 4.2 Official DuoAttention E2E oracle

固定：

```text
同一 model revision
同一 pattern
同一二值化方式
同一 sink/recent
同一 tokenizer
temperature=0
```

比较：

- 短于 window 的 prompt：DuoKV 应接近 FullKV；
- 长于 window 的 prompt：DuoKV 与 Official Duo 的首 token logits/top-1；
- 20 条固定 prompt 的 greedy 输出；
- NIAH 小样本的命中位置。

目标是 greedy 序列一致；如果由于 kernel 浮点顺序不能逐 token 完全一致，最低 Gate 为首 token top-1 一致率不低于 95%，且 logit cosine/最大误差证明没有系统性偏差。任何从第一个 token 开始的大面积分叉都必须定位。

### 调试顺序

```text
pattern shape
→ KV-head partition
→ GQA Q-head expansion
→ restore indices
→ sink/recent 逻辑顺序
→ full_to_comp mapping
→ chunked extend metadata
→ decode ring overwrite
```

### Gate 4

Tensor reference 全部通过，且 E2E 没有系统性分叉，才允许跑容量和吞吐。

---

## Phase 5：物理双池与容量生死 Gate

时间：1 天。

### 要验证什么

1. Compact heads 没有完整历史 tensor；
2. 实际 pool bytes 与预算公式一致；
3. compact chunk 数等于 `max_running_requests`；
4. Full Pool 扩大能转化成 `max_total_tokens`；
5. compact pool 耗尽不会静默返回位置 0 并污染 dummy slot。

### 必做测试

```text
test_pool_tensor_shapes
test_pool_byte_accounting
test_comp_chunk_count
test_sink_slot_mapping
test_recent_ring_mapping
test_ring_wraparound
test_comp_pool_exhaustion_fails_fast
test_all_full_degenerate_path
test_all_compact_rejected_or_special_cased
```

### 容量实验

固定 GPU、模型、dtype、memory fraction、CUDA Graph 状态和 `max_running_requests`，比较：

```text
FullKV max_total_tokens = T0
DuoKV  max_total_tokens = Tf
```

记录：

```text
predicted Tf
actual Tf
full pool bytes
compact pool bytes
mapping/metadata bytes
total KV bytes
```

实际容量增益应达到理论预算增益的至少 90%；若差距更大，必须解释 padding、metadata 或预留显存，而不是只报告 `nvidia-smi`。

### Gate 5：项目生死线

必须同时成立：

- 物理 KV tensor bytes 下降或在相同 budget 下 Full Pool 扩大；
- `Tf > T0`（存在 Compact heads 时）；
- 实际容量趋势与 Full-head ratio 单调一致。

如果只改变 attention 计算但 Compact heads 仍保存完整 KV，项目立即降级，不能叫 KV Cache serving backend。

---

## Phase 6：Request 生命周期与 Continuous Batching

时间：1 天。

### 单请求测试

```text
4K → decode 128
8K → decode 256
16K → decode 256
32K → decode 256（显存允许时）
```

每次完成后检查：

```text
full available == initial full capacity
comp free chunks == initial chunk count
mapping 中已释放 full loc 全部归零
```

### 并发测试

构造混合长度请求：

```text
2K / 4K / 8K / 16K
```

循环执行：

```text
enter → extend → decode → finish → free → slot reuse
```

至少覆盖：

- 两个 request 交错 decode；
- 短请求先完成、长请求继续；
- request slot 重用；
- compact ring 多次 wrap；
- 100、500、1000 个混合请求；
- free-group 批量释放；
- 达到 `max_running_requests` 后的新请求排队，而不是 compact pool 越界。

### Gate 6

连续 1000 个混合请求：

- 无 crash、NaN、串 KV；
- Full/Compact allocator 恢复初始状态；
- 无重复 compact chunk；
- 输出与单请求执行无系统性差异。

如果 eager 通过，再打开 CUDA Graph 重复以上小规模测试。CUDA Graph 不通过时回退 eager，不阻塞 MVP。

---

## Phase 7：正式实验

时间：1.5～2 天。

### Baseline 规则

系统性能只比较：

```text
同一 SGLang commit FullKV
vs
同一 SGLang commit DuoKV-SGLang
```

Official DuoAttention 只用于质量和语义对照，不把其 HF runtime 吞吐与 SGLang 吞吐直接画在同一加速比图中。

### Experiment A：Head ratio 与 KV capacity

测试 effective Full KV-head ratio：

```text
25% / 50% / 75% / 100%
```

记录：

```text
full/compact heads
T0 / predicted Tf / actual Tf
pool bytes
max concurrent requests
```

### Experiment B：Context length

```text
4K / 8K / 16K / 32K
```

记录：

```text
prefill latency
decode TPOT
output tokens/s
peak batch capacity
```

### Experiment C：Concurrency sweep

固定 8K 和 16K 输入，逐步增加：

```text
BS1 / 2 / 4 / 8 / 16 / 32 / ...
```

直到 FullKV OOM/无法准入，而 DuoKV 仍能服务。这个实验比单请求 latency 更能证明项目价值。

### Experiment D：Online serving

至少两个负载：

- memory-light：短上下文、低 QPS；
- memory-bound：长上下文、高并发。

记录：

```text
request/s
input/output tok/s
P50/P95 TTFT
P50/P95 TPOT
P50/P95 E2E latency
max running requests
retracted/queued requests
```

### Experiment E：Quality

至少：

```text
NIAH：8K/16K/32K，多 depth
LongBench：2～3 个 retrieval/QA 子任务
```

对比：

```text
FullKV
Official DuoAttention
DuoKV-SGLang
```

### 实验纪律

- 每个性能点 warmup 后重复至少 3 次；
- 报 median，尾延迟报 P95；
- FullKV 与 DuoKV 使用相同 CUDA Graph 状态；
- 保存完整命令、commit、pattern hash 和原始 CSV；
- 不提前承诺吞吐提升，容量提升是主指标；
- 如果 KV capacity 明显提升而单请求吞吐只提升 0～10%，项目仍成立，应定位 dual dispatch、gather 和 kernel launch 开销。

### 最终图表

1. `Max KV Token Capacity vs Effective Full KV-head Ratio`；
2. `Max Concurrent Requests vs Context Length`；
3. `Throughput vs Concurrency`；
4. `Long-context Accuracy vs KV Memory Ratio`；
5. 可选：`TTFT/TPOT P95 vs Offered Load`。

---

## Phase 8：项目整理与交付

时间：0.5 天。

### 必须保存的证据

```text
README.md
docs/headkv/env.md
docs/headkv/architecture.md
docs/headkv/rlkv_callgraph.md
docs/headkv/known_limitations.md
artifacts/correctness.log
artifacts/allocator_invariant.log
results/capacity.csv
results/throughput.csv
results/online_serving.csv
results/quality.csv
figures/*.png
```

README 必须区分：

- 复用了 RLKV 的哪些 runtime 组件；
- 新增了哪些 policy abstraction 和 DuoAttention 适配；
- 修改了哪些容量/生命周期问题；
- 已经验证的模型和场景；
- 尚未支持的 current main、Prefix Cache、TP 和 speculative decoding。

---

## 7. 可选升级：迁移 current SGLang main

只有 Gate 0～6 全部通过后启动。预计 2～4 天，不计入 MVP。

### current main 迁移框架

需要重新定位四个接入点：

```text
ServerArgs/config validation
→ memory pool registry/factory
→ allocator component
→ attention backend setup factory
```

### 能复用 current Full/SWA 的部分

- dual allocation 的事务/rollback 思想；
- logical full location → secondary location translation API；
- pool registry 与 backend capability 检查；
- CUDA Graph 中保存 secondary write-location metadata 的方式；
- allocator/cache invariant 测试方式。

### 不能直接复用的部分

- layer-level `SWAKVPool` 的 tensor layout；
- Full/SWA layer mapping；
- SWA Radix tombstone 语义；
- 给每个 logical token 同时分配 Full 与 SWA slot 的策略。

HeadKV 的 Compact Pool 是“每个 request 一个固定 ring chunk”，而不是“所有 token 都有第二套 SWA location”。因此 current-main port 应迁移 RLKV 的 request-level compact chunk 语义，不能硬套 layer-level SWA allocator。

### current-main Gate

两天内至少完成：

- HeadPolicy 能加载；
- pool/allocator 能初始化；
- tensor-level backend smoke；
- physical pool shape 正确。

两天仍无法打通端到端 decode，就停止迁移，保留已完成的 v0.5.2 DuoKV-SGLang，不影响项目交付。

---

## 8. 八天执行日历

| 日期 | 主任务 | 当天必须交付 |
| --- | --- | --- |
| Day 1 上午 | 环境冻结、FullKV 与 Official Duo smoke | `env.md`、两份 smoke log |
| Day 1 下午 | RLKV serving 调用链 | `rlkv_callgraph.md`、生命周期图 |
| Day 2 | HeadPolicy、Duo loader、GQA/TP partition | policy/partition 单测全过 |
| Day 3 | ServerArgs、ModelRunner、预算模型接入 | 双池成功初始化、tensor shape 日志 |
| Day 4 | Tensor reference、E2E correctness | correctness Gate 通过 |
| Day 5 | Pool bytes、capacity、allocator 生死 Gate | `Tf>T0`，容量公式对齐 |
| Day 6 | request lifecycle、continuous batching | 1000 请求 invariant log |
| Day 7 | capacity/context/concurrency/online 实验 | 原始 CSV |
| Day 8 | NIAH/LongBench、图表、README | 完整证据链与项目总结 |

如果 Day 5 仍不能证明物理容量提升，停止性能实验，优先修双池预算和 allocator；如果修复一天仍失败，项目降级，不再投入 current-main port。

---

## 9. 止损矩阵

| 风险 | 最大投入 | 止损动作 |
| --- | ---: | --- |
| RLKV fork 环境不稳定 | 0.5 天 | 固定官方依赖和已验证 commit |
| Official Duo 模型/pattern 不匹配 | 0.5 天 | 切换官方明确支持的模型 revision |
| Pattern 维度与 KV heads 不一致 | 2 小时 | 禁止猜测映射，换正确 pattern |
| GQA effective Full ratio 过高 | 2 小时 | 保留 GQA 质量实验，增加 MHA 次模型展示容量上限 |
| Tensor correctness 不通过 | 1 天 | 不进入 allocator/性能阶段 |
| Compact pool 静默耗尽 | 0.5 天 | 增加显式 capacity invariant 和 fail-fast |
| 物理 tensor 未缩短 | 0.5 天 | 判定项目核心失败，停止包装成 KV backend |
| `max_total_tokens` 未提升 | 1 天 | 检查预算、metadata、`max_running_requests` 和重复 buffer |
| Continuous batching 泄漏/重复释放 | 1 天 | 关闭 CUDA Graph，最小化到两请求 interleave |
| CUDA Graph 不稳定 | 0.5 天 | MVP 使用 eager |
| Prefix Cache 不兼容 | 0 天 | MVP 保持关闭 |
| current-main port 卡住 | 2 天 | 停止迁移，交付 v0.5.2 DuoKV-SGLang |
| 单请求吞吐不涨 | 0.5 天定位 | 项目定位为 capacity-oriented serving，不伪造加速故事 |

---

## 10. 最终成功等级

### A 级：可作为秋招主项目

- DuoAttention policy 已从 runtime 解耦；
- 真实 Full/Compact 物理双池；
- `max_total_tokens` 和最大并发提升；
- continuous batching 无泄漏；
- NIAH/LongBench 质量可接受；
- 完整容量、吞吐和尾延迟实验。

项目名：

```text
DuoKV-SGLang：基于 Head Reallocation 的异构 KV Cache Serving
```

### S 级：升级为 HeadKV-SGLang

在 A 级基础上满足其一：

- 同一 runtime 同时支持 DuoAttentionPolicy 与 RLKVPolicy；或
- 完成 current SGLang main 的端到端迁移。

项目名：

```text
HeadKV-SGLang：Policy-decoupled Head-wise Heterogeneous KV Backend
```

### C 级：不能作为 KV Cache 主项目

只完成 mask loader、head partition 或双 attention computation，但 Compact heads 仍保存完整历史。此时只能写“Attention Backend Adaptation”，不能声称 KV capacity 优化。

---

## 11. 项目最终叙事

推荐的技术故事是：

```text
DuoAttention 和 RLKV 虽然用不同方法识别关键 heads，
但部署阶段都归约为 KV-head 的 Full/Compact 两种生命周期。
RLKV 已在 SGLang v0.5.2 验证了真实双池 serving，
因此先将其算法专属 mask loader 与 runtime 解耦，
接入 DuoAttention 官方 pattern，
验证 physical KV capacity、continuous batching 和质量收益；
在 MVP 成立后，再研究如何把该抽象迁移到 current SGLang。
```

这比“直接把 current SGLang 的 layer-level SWA 改成 head-level”更可信，也有明确的阶段成果和止损点。

---

## 12. 参考实现

- RLKV：[Kurt232/RLKV](https://github.com/Kurt232/RLKV)
- RLKV SGLang fork：[Kurt232/rlkv-sglang-v0.5.2](https://github.com/Kurt232/rlkv-sglang-v0.5.2)
- DuoAttention：[mit-han-lab/duo-attention](https://github.com/mit-han-lab/duo-attention)
- SGLang：[sgl-project/sglang](https://github.com/sgl-project/sglang)

