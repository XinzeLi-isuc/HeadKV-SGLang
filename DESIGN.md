# HeadKV-SGLang (DuoKV-SGLang) 技术设计文档

> 前置文档:`HeadKV-SGLang_修订版项目计划书.md`(同目录)。本文档把计划书冻结为可编码的
> 技术规格:接口签名、数据流、预算模型、改动点清单、测试矩阵、实验协议与工作量账本。
> 设计原则:先跑通 RLKV v0.5.2 fork 的 DuoKV MVP(Gate 0~6),再评估 current-main 迁移。

---

## 1. 设计输入与事实基线(2026-08-13 本地实测)

### 1.1 本地资产盘点

| 资产 | 路径 | 状态 |
| --- | --- | --- |
| RLKV 官方 repo(训练/分析) | `~/rlkv` | 已克隆,含 `head_dist/` 多模型 pattern |
| RLKV SGLang fork(v0.5.2) | `~/rlkv/sglang` | commit `973b5e41`,remote=`Kurt232/rlkv-sglang-v0.5.2` |
| DuoAttention 官方实现 | `~/duo-attention-ref` | 含 `attn_patterns/` 与 `duo_attn/` |
| 主模型 pattern | `~/duo-attention-ref/attn_patterns/Meta-Llama-3.1-8B-Instruct/lr=0.02-reg=0.05-ctx=1000_128000-multi_passkey10/` | 32 行 × 8 列,值域已 clip [0,1] |
| 主模型 config | 同上目录 `config.json` | `sink_size=128, recent_size=256, threshold=0.5` |
| 模型权重 | `~/.cache/modelscope/.../Meta-Llama-3.1-8B-Instruct`(与 DuoAttention-Serve 同源) | 待 Phase 0 校验 revision |
| 备份 pattern | `~/rlkv/head_dist/duo_attn/{Llama-3.1-8B-Inst,Llama-3.1-8B-R1,Qwen-3-4B-Thinking,...}` | 备选模型用 |

### 1.2 RLKV fork 现状(关键代码事实,设计以此对齐)

- 入口参数(`python/sglang/srt/server_args.py`):
  - L331-333:`sink_window_size=16`、`recent_window_size=32`、`adapter_load_path=None`
  - L337-338:`enable_rlkv_inference=False`、`rlkv_sparsity=0.5`
  - L748-749:`max_running_requests=None` 时默认置 48
- mask 加载(`model_runner.py::_load_rlkv_head_masks`,L1737-1805):
  - 读 `adapter_weights.tsv`,fallback `full_attention_heads.tsv`
  - `np.loadtxt` → `clip(0,1)` → **加随机微扰** `np.random.uniform(0,1e-6)`(不可复现,须改)
  - sparsity 语义:`threshold = np.quantile(scores, sparsity)`,`mask = (scores >= threshold)`
  - TP shard:`start = tp_rank * num_kv_heads_per_tp` 切片,产物 `{layer_id: Tensor[TP局部kv头]}`(float32)
- 双池初始化(`model_runner.py` L1552-1597):
  - `budget = max_total_num_tokens * total_heads`(即 T0×(F+C))
  - `size_comp = max_num_reqs * window`;`comp_cost = size_comp * total_comp`
  - `size_full = max((budget - comp_cost) // total_full, max_total_num_tokens)`(**注意现状带 T0 下限**,设计须统一)
  - 覆写 `self.max_total_num_tokens = size_full`;分配器 `HeadReallocAllocator(max_total_num_tokens, size_comp, window_size=sink+recent, ...)`(L1638-1646)
- 存储(`mem_cache/memory_pool.py::HeadReallocKVPool`,L892):
  - 每层两个 buffer:full `[size_full+1, n_full, head_dim]`、comp `[size_comp+1, n_comp, head_dim]`
  - `full_head_indices/comp_head_indices` 由 mask 派生;`set_kv_buffer` 用 `_fused_kv_write` Triton kernel 一次写双池
  - `full_to_comp_mapping` 由 allocator 持有(L329-335),backend 填充
- 分配器(`mem_cache/allocator.py::HeadReallocAllocator`,L290):
  - full pool 走标准 `TokenToKVPoolAllocator`;comp pool 按 request 分 chunk
  - `max_comp_chunks = size_comp // window_size`;`alloc_comp_window()` 返回 `1 + chunk_id*window`(0 表示耗尽)
  - `free()` 从 `full_to_comp_mapping[free_index]` 反推 comp_base 释放 chunk(L367-384)
- 后端(`layers/attention/head_realloc_backend.py`,1111 行):
  - `__init__` L71:`sink_window_size` / `local_window_size`(recent)/ `window_size = sink+recent`
  - `_get_comp_base(req_pool_idx)` L194;`_update_comp_mapping_decode` L217;`_update_comp_mapping_extend` L266
  - `forward_decode` L718 / `forward_extend` L823:full/comp Q 分组,分别用 `decode_attention_fwd`(full 走 `kv_indptr/kv_indices`,comp 走 `window_kv_indptr/window_kv_indices`)
  - sink 位置:`positions < sink` 直接写;recent 环形:`sink + (positions - sink) % recent`

### 1.3 DuoAttention 官方语义(已核实,设计须对齐)

- `duo_attn/utils.py::sparsify_attention_heads`(L353):
  - 也加随机微扰破同分(官方同样不可复现 → 本项目改用确定性 top-k,作为改进点)
  - `sparsity` 模式:`threshold = np.quantile(scores, sparsity)`,`mask = (scores >= threshold)` → 保留 top-(1-sparsity) 比例
  - `threshold` 模式:`mask = (scores >= threshold)`,默认 0.5
- 训练期 `clamp_(x, 0, 1)`;但早期 step 保存的 tsv 可能含负值 → 加载时仍须 clip
- 官方 pattern 对 Llama-3.1-8B 已是 KV-head 粒度(8 列 = num_kv_heads),**不得二次 OR 聚合**
- 官方 config 默认 `sink_size=128, recent_size=256`(vs RLKV fork 默认 16/32,差异显著)

### 1.4 设计必须消除的现状缺陷(计划书修订表的落地)

| # | 缺陷 | 设计对策 |
| --- | --- | --- |
| D1 | 随机微扰破同分,不可复现 | 确定性稳定排序 `(score, layer_id, head_id)` 取 top-k |
| D2 | RLKV loader 只认 adapter/full_attention_heads.tsv | `HeadPolicy` 抽象,加载器与 runtime 解耦 |
| D3 | 默认 window 16/32 与 Duo 官方 128/256 不一致 | 默认读 pattern 目录 `config.json`,命令行显式覆盖 |
| D4 | 预算公式带 `max(..., T0)` 下限,与计划书公式不一致 | 统一为计划书公式,边界显式报错 |
| D5 | `max_running_requests` 默认 48 可能污染 comp 预算 | HeadKV 模式强制显式指定 |
| D6 | `--enable-rlkv-inference` 直达 ModelRunner,policy 耦合 | 保留为兼容入口,内部转为 HeadPolicy |

## 2. 总体架构

### 2.1 分层与模块

```text
CLI / ServerArgs (--enable-headkv ...)
        │
        ▼
HeadKVConfig.validate()            [headkv/config.py]  参数合法性 + window 优先级
        │
        ▼
HeadPolicy.load_global_kv_mask()   [headkv/policy.py]  返回 [layers, global_kv_heads] bool mask
        │  DuoAttentionPolicy / RLKVPolicy / 人工 mask
        ▼
partition.to_tp_local()            [headkv/partition.py]  global → TP-local(MVP 恒等,TP=1)
        │
        ▼
budget.compute()                   [headkv/budget.py]    Tf / Tc / 边界检查
        │
        ▼
HeadReallocKVPool + HeadReallocAllocator   [mem_cache/]  物理双池(复用,修边界 bug)
        │
        ▼
HeadReallocAttnBackend             [layers/attention/]  full/comp 双路 attention(复用,通用化)
        │
        ▼
Triton kernels(_fused_kv_write / decode_attention_fwd / window buffer)
```

新增代码全部收敛在 `python/sglang/srt/headkv/`(纯算法与配置,零 SGLang 内部依赖,
便于单测与 future current-main port);对现有文件的修改控制在最小 diff。

### 2.2 数据流

**启动序列**(ModelRunner.init 阶段,替换现有 L1552-1597 的 RLKV 分支):

```text
ServerArgs
→ HeadKVConfig.from_server_args()        # 校验 + 解析优先级
→ policy = HeadPolicy.create(cfg)        # duo | rlkv | manual
→ global_mask = policy.load_global_kv_mask(model_config)   # [L, G_kv]
→ tp_mask = partition.to_tp_local(global_mask, tp_rank, tp_size)  # MVP: TP=1 恒等
→ budget = budget.compute(T0, F, C, R, V)   # 计划书 §3 公式,含边界
→ HeadReallocKVPool(size_full=Tf, size_comp=Tc, head_masks=tp_mask, ...)
→ HeadReallocAllocator(Tf, Tc, window_size=V, ...)
→ HeadReallocAttnBackend(model_runner, tp_mask)
```

**每请求生命周期**(与 Phase 1 逆向结果一致,见 §4):

```text
enter → extend(alloc full loc + alloc comp window + 写 sink/recent 映射)
      → decode(每次写 full loc + 环形覆盖 comp recent 槽)
      → finish → free(full loc 归还 + 由 mapping 反推 comp base 归还 + mapping 清零)
```

### 2.3 组件映射(复用 / 修改 / 新增)

| 组件 | 动作 | 说明 |
| --- | --- | --- |
| `HeadReallocKVPool` | 复用+小修 | 修命名/边界 bug(§5.4) |
| `HeadReallocAllocator` | 复用+小修 | 同上 |
| `HeadReallocAttnBackend` | 复用+小修 | 通用化命名,不动 attention 算法 |
| `_load_rlkv_head_masks` | 移除 | 由 `RLKVPolicy` 取代 |
| `headkv/` 包 | 新增 | config/policy/partition/budget |

---

## 3. 接口规格(代码级)

### 3.1 HeadKVConfig(`headkv/config.py`)

```python
@dataclass
class HeadKVConfig:
    enable: bool = False
    policy: str = "duo"                    # "duo" | "rlkv" | "manual"
    pattern_path: Optional[str] = None     # 目录,须含 full_attention_heads.tsv + config.json
    full_head_ratio: Optional[float] = None   # 与 threshold 二选一
    threshold: Optional[float] = None      # 官方语义: score >= T 为 full
    sink_size: Optional[int] = None        # 显式覆盖
    recent_size: Optional[int] = None      # 显式覆盖
    max_running_requests: Optional[int] = None  # HeadKV 模式必填

    @classmethod
    def from_server_args(cls, args) -> "HeadKVConfig": ...
    def validate(self) -> None: ...
```

校验规则(全部 fail-fast):
- `enable=True` 时 `max_running_requests` 必须显式给出(拒绝默认 48 静默生效)
- `full_head_ratio` 与 `threshold` 同时给出 → 报错;都未给出 → 用 pattern `config.json`
  的 `threshold`(官方默认 0.5)
- `0 < full_head_ratio <= 1`;`threshold` 可为负(官方允许负阈值 = 全 full)
- pattern 目录缺 `full_attention_heads.tsv` → 报错(不 fallback 到 RLKV adapter)

window 优先级(计划书 §2 修订,禁止静默使用 RLKV 默认):

```text
CLI --headkv-sink-size / --headkv-recent-size
  > config.json 的 deploy_sink_size / deploy_recent_size
  > config.json 的 sink_size / recent_size
  > 报错(不落到 RLKV fork 的 16/32 默认)
```

### 3.2 HeadPolicy(`headkv/policy.py`)

```python
class HeadPolicy(abc.ABC):
    """Runtime 只消费 mask + window,不关心来源。"""

    @abc.abstractmethod
    def load_global_kv_mask(self, model_config) -> torch.Tensor:
        """bool mask, shape=[num_layers, global_num_kv_heads]。True=Full, False=Compact。
        返回前必须通过 dimension_check(§3.3)。"""

    @abc.abstractmethod
    def sink_size(self) -> int: ...
    @abc.abstractmethod
    def recent_size(self) -> int: ...

    @classmethod
    def create(cls, cfg: HeadKVConfig) -> "HeadPolicy":
        return {"duo": DuoAttentionPolicy, "rlkv": RLKVPolicy,
                "manual": ManualPolicy}[cfg.policy](cfg)
```

**核心不变量**:runtime 中任何地方不得出现 `if policy == "duo"` 之类分支;policy 差异
只存在于 `load_global_kv_mask` 与 window 解析。

### 3.3 DuoAttentionPolicy(`headkv/duo_policy.py`)

```python
class DuoAttentionPolicy(HeadPolicy):
    def __init__(self, cfg: HeadKVConfig):
        self.cfg = cfg
        self._scores: Optional[np.ndarray] = None   # [L, G_kv], clip 后
        self._sink = self._recent = None

    def load_global_kv_mask(self, model_config) -> torch.Tensor:
        self._load_pattern()                # tsv + config.json
        self._validate_shape(model_config)  # GQA 规则,见下
        return self._binarize()             # 确定性 top-k 或 threshold
```

加载与二值化流程(确定性,无任何随机源):

```text
np.loadtxt(full_attention_heads.tsv, delimiter="\t")
→ np.clip(scores, 0, 1)
→ shape 校验(见下)
→ 模式 A(full_head_ratio R):  稳定 top-k
      k_layer = round(R * G_kv)
      tie-break: 按 (score, layer_id, head_id) 字典序稳定排序,取前 k
→ 模式 B(threshold T):        mask = (scores >= T)
→ bool mask [L, G_kv] (True=Full)
```

**GQA 维度校验**(计划书 §2 修订,严格执行):
- 列数 == `model_config.num_kv_heads`(global)→ 直接使用,**禁止二次 OR 聚合**(D1.3)
- 列数 == `model_config.num_q_heads` → 按共享 KV group 做 OR:第 g 个 KV head 对应
  Q heads `[g*n_q_per_kv, (g+1)*n_q_per_kv)`,任一为 True 则 KV head 为 Full
- 其他维度 → `ValueError`,禁止猜测映射

配置优先级:
```text
cfg.sink_size / cfg.recent_size(CLI)
  > config.json deploy_sink_size / deploy_recent_size
  > config.json sink_size / recent_size
  > ValueError
```

日志必须输出:pattern path、shape、F/C 计数、nominal ratio、effective ratio
(= 实际 Full 头数 / 总头数)、sink/recent/window、二值化模式与参数。

### 3.4 RLKVPolicy(`headkv/rlkv_policy.py`,兼容)

- 包装原 `_load_rlkv_head_masks` 语义:读 `adapter_weights.tsv`(fallback
  `full_attention_heads.tsv`)、clip、sparsity-quantile 二值化
- **去掉随机微扰**,改用与 Duo 相同的确定性 tie-break
- `sink_size/recent_size` 显式传入(不再读 fork 默认)
- 用途:兼容 `--enable-rlkv-inference` 老入口 + 后续同一 runtime 双 policy 对照(S 级)

### 3.5 Partition(`headkv/partition.py`)

```python
def to_tp_local(global_mask: torch.Tensor, tp_rank: int, tp_size: int,
                num_kv_heads_per_tp: int) -> Dict[int, torch.Tensor]:
    """返回 {layer_id: float32 tensor[TP局部kv头]}。与 fork L1783-1797 切片语义一致。
    MVP 仅 TP=1(恒等映射),接口保持通用。"""
```

不变量:`to_tp_local` 是**完备且不相交**的划分:各 rank mask 并集 == global mask,
交集为空。单测覆盖。

### 3.6 Budget(`headkv/budget.py`)

输入:T0(FullKV profiling 的 baseline token capacity)、F(全层 Full 头数之和)、
C(全层 Compact 头数之和)、R(max_running_requests)、V = sink+recent。

```python
def compute(T0, F, C, R, V) -> Budget:
    Tc = R * V
    # 保持 KV byte budget 近似不变: T0*(F+C) = Tf*F + Tc*C
    Tf = (T0 * (F + C) - Tc * C) // F
    # 边界:
    #  F == 0  → 拒绝 serving 配置(或全 streaming 专用路径,MVP 直接报错)
    #  C == 0  → 退化为 FullKV,不分配 comp pool(Tf = T0)
    #  Tc*C >= T0*(F+C) → 配置不可能,启动报错
    #  R 未显式 → HeadKV 启动报错(§3.1)
```

与 fork 现状(L1571-1575)的差异:现状 `size_full = max(..., T0)` 带下限;设计统一为
计划书公式并显式记录。若某配置下 `Tf < T0`(当 `Tc*C > T0*C` 即 `R*V > T0` 时,实际
几乎不会发生,因 R*V 远小于 T0),如实打印并给出解释,不静默 clamp。

Gate 3 启动日志必须含:policy/path、L/Q/KV heads、F/C 计数、sink/recent/window、
R、T0、Tf、Tc、predicted capacity gain = Tf/T0。

## 4. 双池生命周期设计

### 4.1 地址空间与 ring 布局

```text
Full pool:  [Tf + 1] 槽,0 为 dummy,1..Tf 由 full_allocator 管理(标准 alloc/free)
Comp pool:  [Tc + 1] 槽,0 为 dummy,按 request 分 chunk:
            chunk_id ∈ [0, max_comp_chunks), comp_base = 1 + chunk_id * V
            chunk 内部布局(按 token 位置 p,seq_len = L):
              p < sink                → 槽 sink + p
              sink <= p < L-recent    → 不写(该 token 的 comp KV 不落盘)
              p >= L-recent           → 槽 sink + (p - (L-recent))   # recent 环形区
            等价于 fork `_update_comp_mapping_extend`(L266)语义:
              is_recent = (p >= L - recent); 槽 = sink + (p - (L - recent)) % recent
```

要点:
- **sink 区只在 extend 时写入**(p < sink);decode 阶段不碰 sink
- **recent 区环形覆盖**:decode 每步把新 token 写到 `sink + (p - (L-recent)) % recent` 槽,
  即恒覆盖最旧 recent token;`p < sink` 的 decode 步骤(seq 很短时)写 sink 区
- `full_to_comp_mapping[full_loc] = comp_loc` 逐 token 维护,是 free 反推的唯一依据
- 一个 request 的 comp chunk 内,同一 full_loc 永不重复写(每 token 唯一 full loc → 唯一 comp 槽)

### 4.2 extend 路径(新请求 / 续写)

```text
1. full_allocator.alloc(extend_len) → full locs(连续)
2. backend._get_comp_base(req_pool_idx):查 per-request 已有 comp_base,无则 alloc_comp_window()
   (fork L194-215:已有则从 mapping 反推,保持同一 chunk)
3. set_kv_buffer(layer, out_cache_loc, k, v):
   _fused_kv_write 按 full/comp head indices 分流写两个 buffer
4. _update_comp_mapping_extend(out_cache_loc, positions, seq_lens):
   仅 in_window(sink ∪ recent)位置写 full_to_comp_mapping;窗口外位置保持 0
5. init_forward_metadata 构造 window_kv_indptr/window_kv_indices:
   window_kv_lens = min(L, V);indptr = cumsum;indices 按 sink/recent 地址映射
   (fork L381-430)
```

chunked extend:同一 request 多次 extend 时,`_get_comp_base` 复用同一 chunk;
comp 槽按**新 seq_len** 重新计算(recent 环形滚动),旧 recent 槽被覆盖,语义与
一次性 extend 一致(需单测验证 chunked == 一次性的 comp 落盘结果)。

### 4.3 decode 路径

```text
1. set_kv_buffer(layer, out_cache_loc, k, v):写 full 池新槽
2. _update_comp_mapping_decode(out_cache_loc, positions, ...):
   p < sink        → 写 sink 槽(仅极短序列出现)
   否则            → 写 recent 环形槽 sink + (p - sink) % recent
3. forward_decode:full heads 走 kv_indptr/kv_indices(全历史);
   comp heads 走 window_kv_indptr/window_kv_indices(仅窗口)
```

### 4.4 free 路径(计划书 Phase 1 问题 8)

fork 现状(L367-384)已实现反推释放,设计保留并加固:

```text
free(free_index):
  1. full_allocator.free(free_index)                      # full 槽归还
  2. comp_indices = full_to_comp_mapping[free_index]
     non_zero = comp_indices[comp_indices > 0]
     comp_bases = (non_zero - 1) // V * V + 1             # 同一 chunk 归一到 base
     unique_bases 逐个 free_comp_window(base)             # 防重复释放
  3. full_to_comp_mapping[free_index] = 0                 # mapping 清零
```

free-group 批量释放:fork 用 `free_group.append` 延迟,`flush` 时统一走上述逻辑;
设计保留该机制,单测覆盖"同一 chunk 多 loc 批量释放只 free 一次"。

### 4.5 并发交错与 fail-fast

- 多 request 交错:comp chunk 按 request 隔离(alloc_comp_window 弹栈),天然无串扰;
  full loc 由标准 allocator 管理
- **comp 池耗尽**:`alloc_comp_window()` 返回 0 → 当前代码**静默返回 0 并可能污染
  dummy slot**(计划书 Phase 5 要点 5)→ 设计改为 fail-fast:抛 `RuntimeError`
  并打印 comp 池状态(`comp_chunks_available/max_comp_chunks`)
- **full 池耗尽**:沿用引擎现有 admission 控制(available_size 检查,请求排队)
- 顺序不变量:同一 request 的 extend/decode 按 token 顺序执行,comp 槽计算只依赖
  `(p, L)` 与当前 mapping,无跨 request 依赖

---

## 5. 改动点清单(按文件)

### 5.1 新增 `python/sglang/srt/headkv/`(约 5 个文件,~600 行)

| 文件 | 内容 | 依赖 |
| --- | --- | --- |
| `__init__.py` | 空 | - |
| `config.py` | `HeadKVConfig` + 校验 + window 优先级解析 | 仅 stdlib |
| `policy.py` | `HeadPolicy` 抽象 + `create` 工厂 + `ManualPolicy` | numpy |
| `duo_policy.py` | tsv/config 加载、确定性二值化、GQA 校验 | numpy |
| `rlkv_policy.py` | 兼容包装(去随机微扰) | numpy |
| `partition.py` | `to_tp_local` | torch |
| `budget.py` | `compute()` + 边界 | 纯算术 |

### 5.2 `server_args.py`(修改)

- 新增参数:`--enable-headkv`、`--headkv-policy`、`--headkv-pattern-path`、
  `--headkv-full-head-ratio`、`--headkv-threshold`、`--headkv-sink-size`、
  `--headkv-recent-size`(类型/默认见 §3.1)
- 保留 `--enable-rlkv-inference` / `--rlkv-sparsity` / `--adapter-load-path` /
  `--sink-window-size` / `--recent-window-size` 为兼容入口(不删除,避免破坏老脚本)
- 二者互斥校验:`enable_headkv` 与 `enable_rlkv_inference` 同时为 True → 报错

### 5.3 `model_runner.py`(修改,L1552-1597 区域)

- `enable_rlkv_inference` 分支改为通用 `enable_headkv` 分支,流程见 §2.2
- 删除 `_load_rlkv_head_masks`(逻辑迁往 `RLKVPolicy`);`rlkv_head_masks` 属性改名
  `headkv_head_masks`(兼容 getter 保留)
- backend 选择(L1858-1867):统一 `HeadReallocAttnBackend(model_runner, headkv_head_masks)`

### 5.4 `memory_pool.py` / `allocator.py`(小修清单)

- `HeadReallocKVPool.set_kv_buffer`:`layer_id_override` 分支正确性复核(L1001-1010),
  确认 chunked extend 传参一致
- `HeadReallocAllocator.free`:防重复释放已由 unique 保证,补 `free_comp_window`
  越界保护(L356-361)
- `alloc_comp_window` 返回 0 的调用点改 fail-fast(§4.5)
- 命名通用化:`rlkv_head_masks` → `headkv_head_masks`(仅重命名,不动语义)

### 5.5 `head_realloc_backend.py`(小修清单)

- `_get_comp_base`(L194):确认"已有 mapping 反推"分支与 chunked extend 兼容
- comp 耗尽 fail-fast(§4.5)
- CUDA Graph 相关 metadata(window buffer)MVP 阶段保持现状,不新增逻辑

### 5.6 测试与文档目录(计划书 §5 已列,补充)

```text
test/srt/headkv/           # 8 个测试文件,见 §6.1
docs/headkv/               # env/architecture/callgraph/protocol/limitations
benchmarks/headkv/         # run_capacity/run_offline_throughput/run_online_serving/run_niah + configs/
```

## 6. 测试设计

### 6.1 单元测试矩阵(`test/srt/headkv/`,全部 CPU 可跑,不进 GPU)

| 文件 | 用例 | 断言要点 |
| --- | --- | --- |
| `test_duo_policy.py` | test_load_official_duo_pattern | 真实 pattern 加载后 shape=[32,8],值域 [0,1] |
| | test_pattern_shape_validation | 列数≠kv/q heads → ValueError |
| | test_deterministic_topk | 同输入两次运行 mask 完全一致(回归 D1) |
| | test_topk_tie_break | 构造同分,验证 (score,layer,head) 序稳定 |
| | test_window_config_priority | CLI > deploy_* > config.json > 报错(回归 D3) |
| | test_gqa_kv_pattern_no_second_aggregation | 8 列 pattern 原样输出(回归 D1.3) |
| | test_qhead_pattern_or_aggregation | 32 列 pattern OR 后 [32,8] 且语义正确 |
| | test_negative_threshold_all_full | T=-1 → 全 full(C=0 退化路径) |
| `test_partition.py` | test_tp1_partition | TP=1 恒等 |
| | test_partition_complete_and_disjoint | TP=2 并集完备、交集为空(接口通用性) |
| `test_budget.py` | test_budget_formula | 构造 (T0,F,C,R,V),验证 Tf/Tc 与公式一致 |
| | test_budget_capacity_gain | F<total → Tf>T0;F=total → Tf=T0 |
| | test_budget_f0_rejected | F=0 → 报错 |
| | test_budget_c0_degenerate | C=0 → Tf=T0,不分配 comp |
| | test_budget_impossible | Tc*C ≥ T0*(F+C) → 报错 |
| | test_budget_r_required | R 缺失 → 报错(回归 D5) |
| `test_headkv_pool.py` | test_pool_tensor_shapes | full/comp buffer 每层 shape 正确 |
| | test_pool_byte_accounting | get_kv_size_bytes == Σ 元素×element_size |
| | test_fused_write_splits_heads | set_kv_buffer 后 full/comp 数据按 indices 正确分流 |
| | test_no_full_history_copy | 不存在 [T0, all_heads, head_dim] 副本(Gate 5 预演) |
| `test_headkv_allocator.py` | test_comp_chunk_count | max_comp_chunks = Tc // V |
| | test_comp_window_alloc_free | alloc→free 循环无泄漏、无重复 |
| | test_free_derives_comp_base | 同一 chunk 多 loc 批量释放只 free 一次 |
| | test_mapping_cleared_on_free | free 后 full_to_comp_mapping 归零 |
| | test_comp_pool_exhaustion_fails_fast | 耗尽后 alloc 抛 RuntimeError |
| | test_all_full_degenerate_path | 全 full mask 下 comp buffer 存在但 size_comp=0 |
| `test_headkv_attention.py` | 见 §6.2 tensor reference(CPU/小 GPU) | |
| `test_headkv_lifecycle.py` | 见 §6.4(小规模 GPU,3 个用例保底) | |

### 6.2 Tensor-level reference(Phase 4.1,纯 PyTorch 实现)

参考实现 `benchmarks/headkv/ref_attention.py`(不进 SGLang 依赖):

```python
def ref_attention(q, k_full, v_full, head_mask, sink, recent):
    """对每个 (layer, head):
       full head → 标准 causal attention(全历史)
       comp head → 只允许访问位置 [0, sink) ∪ [L-recent, L)"""
```

覆盖矩阵:

```text
配置:  all-full / all-compact / 50% mixed / 每层不同 mask / GQA shared KV groups
序列:  L < V / L = V / L > V(至少 L = 2V 触发环形覆盖)
模式:  一次性 extend / chunked extend(2 块)/ 逐 token decode(extend 后)
```

对比方式:同一输入 q/k/v,ref 输出 vs `HeadReallocAttnBackend` 输出;头分组与
restore indices 逐一核对。容差起点 `atol=rtol=1e-2`(FP16);超差先定位数值路径
(不做无意义放宽)。

### 6.3 E2E correctness oracle(Phase 4.2)

固定:同一模型 revision / 同一 pattern / 同一二值化 / 同一 sink·recent / 同一
tokenizer / temperature=0。三方对比:FullKV、DuoKV-SGLang、Official DuoAttention
(仅 oracle,不参与系统性能比较)。

- 短于 window 的 prompt:DuoKV 输出应接近 FullKV(逐 token 一致或 top-1 一致)
- 长于 window 的 prompt:首 token top-1 一致率 ≥ 95%;logit cosine / max-abs-error
  证明无系统性偏差;大面积分叉从首 token 起即定位
- 20 条固定 prompt(`benchmarks/headkv/prompts_correctness.jsonl`)greedy 输出存档
- NIAH 小样本(每长度 20 条)命中位置记录

### 6.4 生命周期与容量 invariant(Phase 5/6,GPU 小规模)

```text
test_sink_slot_mapping / test_recent_ring_mapping / test_ring_wraparound
test_comp_pool_exhaustion_fails_fast
test_lifecycle_single_request:   4K→decode128 后 full available==初始、comp chunks==初始、
                                 mapping 归零
test_lifecycle_mixed_1000:       2K/4K/8K/16K 混合,1000 请求循环
                                 enter→extend→decode→finish→free→slot reuse
                                 (交错 decode、短先完成、ring 多次 wrap、free-group 批量)
```

Gate 6 判定:无 crash / 无 NaN / 无串 KV / allocator 恢复初始态 / 无重复 comp
chunk / 输出与单请求无系统性差异。

### 6.5 测试命令与 Gate 映射

```text
cd ~/rlkv/sglang
python -m pytest test/srt/headkv/ -x -q --ignore=test/srt/headkv/test_headkv_attention.py
   # CPU 单测 → Gate 2
python -m pytest test/srt/headkv/test_headkv_attention.py -x -q   # GPU tensor ref → Gate 4
python -m pytest test/srt/headkv/test_headkv_lifecycle.py -x -q   # GPU 生命周期 → Gate 5/6
```

纪律(用户既定工作流):代码验收优先,pytest+py_compile 全过才上 GPU;禁边跑边修;
内嵌 `python -c` 一律写成脚本文件;显式展示工具输出。

---

## 7. 实验协议

### 7.0 环境冻结(Phase 0,每轮实验前执行并记录)

```bash
git -C ~/rlkv/sglang rev-parse HEAD
python -V
python -c "import torch; print(torch.__version__, torch.version.cuda)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
```

实验期间禁止 `git pull`。conda env 按 fork requirements 固定(与 DuoAttention-Serve
同源 torch 版本,Phase 0 实测后写入 `docs/headkv/env.md`)。

### 7.1 固定运行约束(所有实验)

```text
TP=1 / page_size=1 / Prefix Cache=OFF / CUDA Graph=OFF(MVP)/ Speculative=OFF
temperature=0(正确性)/ dtype 同源(FP16 或 BF16,FullKV 与 DuoKV 一致)
max_running_requests 显式指定并写日志 / 单卡 A6000
```

### 7.2 Experiment A:Head ratio vs KV capacity

```text
full_head_ratio ∈ {0.25, 0.50, 0.75, 1.00}(threshold=0.5 的官方 pattern 另作一组)
记录: F/C、T0 / predicted Tf / actual Tf、pool bytes(full/comp/metadata)、
      max concurrent requests → results/capacity.csv
```

### 7.3 Experiment B:Context length

```text
4K / 8K / 16K / 32K(显存允许)
记录: prefill latency、decode TPOT、output tok/s、peak batch capacity
      → results/throughput.csv
```

### 7.4 Experiment C:Concurrency sweep(项目价值主证据)

```text
固定 8K 与 16K 输入;BS ∈ {1,2,4,8,16,32,...} 递增,
直到 FullKV OOM/无法准入而 DuoKV 仍可服务 → 记录双方 max BS 与对应吞吐
```

### 7.5 Experiment D:Online serving

```text
memory-light(短上下文低 QPS)与 memory-bound(长上下文高并发)两负载
记录: request/s、input/output tok/s、P50/P95 TTFT/TPOT/E2E、
      max running requests、retracted/queued → results/online_serving.csv
```

### 7.6 Experiment E:Quality

```text
NIAH: 8K/16K/32K 多 depth;LongBench: 2~3 个 retrieval/QA 子任务
对比 FullKV / Official DuoAttention / DuoKV-SGLang → results/quality.csv
```

实验纪律:每点 warmup 后 ≥3 次取 median,P95 报尾延迟;FullKV 与 DuoKV 同 CUDA
Graph 状态;存完整命令/commit/pattern hash/原始 CSV。容量是主指标,吞吐不承诺
提升幅度;若容量提升明显而单请求吞吐 0~10%,定位 dual dispatch/gather/launch
开销并如实写进 known_limitations。

最终图表(5 张):capacity vs ratio、max concurrent vs ctx length、throughput vs
concurrency、accuracy vs KV memory ratio、TTFT/TPOT P95 vs load。

## 8. 工作量账本与执行日历

### 8.1 任务 × 人日 × 方差

| 任务 | 人日 | 方差 | 前置 | 产出 |
| --- | ---: | --- | --- | --- |
| T1 环境冻结 + FullKV/Official Duo smoke | 0.5 | ±0.25(依赖版本坑) | - | env.md、两份 smoke log |
| T2 RLKV 调用链逆向(10 问) | 0.5 | ±0.25 | T1 | rlkv_callgraph.md、architecture.md |
| T3 headkv 包:config/policy/duo/partition/budget | 1.0 | ±0.5(官方语义对齐) | T2 | 单测全过(Gate 2) |
| T4 ServerArgs + ModelRunner 接入 | 0.5 | ±0.25(改 fork 边界) | T3 | 双池初始化日志(Gate 3) |
| T5 双池/allocator 小修 + pool 单测 | 0.5 | ±0.25 | T3 | pool/allocator 单测 |
| T6 Tensor reference + 语义对齐 | 1.0 | ±0.5(容差/数值路径) | T4 | ref 全过(Gate 4) |
| T7 E2E correctness(oracle 三方) | 0.5 | ±0.25 | T6 | correctness.log |
| T8 容量 Gate + 生命周期/1000 请求 | 1.0 | ±0.5(泄漏排查) | T7 | invariant log(Gate 5/6) |
| T9 实验 A-E | 1.5 | ±0.5(GPU 排队) | T8 | 5 张图 + 原始 CSV |
| T10 整理交付(README/文档/证据链) | 0.5 | ±0.25 | T9 | 完整证据链 |

合计约 **7.5 人日,方差 ±2.5**;对照计划书 8 天日历可行,但有 Day 5(Gate 5 生死线)
与 Day 7(实验)两个关键路径,建议预留 1 天 buffer。current-main port(2~4 人日)
不计入 MVP。

### 8.2 八天执行日历(细化,含止损点)

| 日期 | 主任务 | 当天 Gate |
| --- | --- | --- |
| Day 1 上午 | T1 | Gate 0(smoke 通过) |
| Day 1 下午 | T2 | Gate 1(生命周期图完整) |
| Day 2 | T3 | Gate 2(mask 确定性 + partition 完备) |
| Day 3 | T4 + T5 | Gate 3(双池 shape 日志,无全量副本) |
| Day 4 | T6 + T7 | Gate 4(E2E 无系统分叉) |
| Day 5 | T8 | **Gate 5(Tf>T0,物理容量成立)** — 失败则停性能实验修预算 |
| Day 6 | T8 续(1000 请求) | Gate 6(invariant 全过) |
| Day 7 | T9 | 原始 CSV 齐全 |
| Day 8 | T10 | 完整交付 |

### 8.3 Gate 检查表(启动每阶段前过一遍)

- [ ] 代码验收:pytest + py_compile 全过,无 pending fix
- [ ] 环境冻结记录存在(commit/python/torch/nvidia-smi)
- [ ] 显式 `--max-running-requests`,日志含 R/T0/Tf/Tc
- [ ] pattern hash 记录(sha256 of tsv)
- [ ] 实验命令完整存档(禁边跑边改参数)

---

## 9. 风险与止损(计划书 §9 增量)

| 风险 | 最大投入 | 止损动作 |
| --- | ---: | --- |
| fork 与 torch 版本不匹配 | 0.5 天 | 按 fork requirements 重装 env;备选:cp -a 现有 env(见记忆) |
| Duo pattern 与模型 revision 不匹配 | 0.5 天 | 换 `~/rlkv/head_dist/duo_attn/` 备份 pattern 或官方明确支持的 revision |
| GQA effective ratio 过高(收益小) | 2 小时 | 保留 GQA 质量结论,加 Mha 次模型(Llama-2-7B-32K)展示上限 |
| 官方二值化语义理解偏差 | 2 小时 | 以 `duo_attn/utils.py::sparsify_attention_heads` 为唯一参照,单测对齐 |
| `Tf` 公式与 fork 现状冲突 | 2 小时 | 以计划书公式为准,差异写进 architecture.md,不静默 clamp |
| comp 池静默耗尽 | 0.5 天 | fail-fast(§4.5),invariant 测试 |
| 物理 tensor 未缩短 | 0.5 天 | 项目降级,停止包装为 KV backend(计划书 C 级) |
| current-main port 卡住 | 2 天 | 停止,交付 v0.5.2 DuoKV-SGLang |
| 并行会话干扰(共享 ~/cake-serve 类似情形) | 0.5 天 | 实验前 `nvidia-smi` 核对显存,杀进程先验证父进程链 |

---

## 10. Git 与交付纪律

- 在 `~/rlkv/sglang` fork 上新建独立分支(如 `feat/headkv-duo`),**不碰 main**;
  本地无 git 全局 config → 提交前设 `user.name/user.email`(惯例
  `lixinze <lixinze@users.noreply.github.com>`)
- 提交按模块拆分(T3 一个提交、T4 一个提交……),**禁大杂烩 commit**,逐 commit 可审
- 模式:先本地 git 先行,项目成熟后再推远端独立仓库(不在原 fork 分支操作)
- 面试/设计内部文档只本地(design 文档不进远端);对外材料禁夸大,数字须实测可追问,
  措辞用"接入/集成/改造",明确基线边界:复用 RLKV v0.5.2 fork 的 head-reallocation
  runtime,个人贡献 = HeadPolicy 抽象 + DuoAttention 确定性接入 + 预算/生命周期修复,
  非原创算法
- README 必须区分:复用了 RLKV 哪些组件 / 新增了哪些抽象 / 修了哪些问题 /
  已验证的模型场景 / 未支持的 current-main、Prefix Cache、TP、speculative

---

## 11. 待确认问题(开工前与用户对齐)

1. conda env 策略:新建独立 env 还是 `cp -a` 复制现有 env(记忆:新建 env 遇
   aau_token None bug)?fork 的 requirements 实测后再定。
2. 主模型权重 revision:沿用 DuoAttention-Serve 的 modelscope 源,还是 HF 直连
   (需代理 127.0.0.1:7883,禁 SSL 校验)?
3. dtype 固定 FP16 还是 BF16(与 FullKV baseline 同源即可,倾向 FP16 对齐官方)?
4. full_head_ratio 默认值:官方 threshold=0.5 的 pattern 实测 effective ratio
   后决定 Experiment A 的默认档位。




