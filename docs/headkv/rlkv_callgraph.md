# RLKV Serving 调用链逆向(rlkv_callgraph.md)

> Phase 1 交付物,2026-08-13。基于 `~/rlkv/sglang` @ 973b5e41 源码逐行核对。
> 所有行号均为当前 commit 的真实行号。

## 总览:一次请求的完整数据流

```text
HTTP /generate
  → TokenizerManager(tokenize)→ ZMQ → Scheduler.event_loop
  → SchedulePolicy(准入:available_size / max_running_requests)
  → prepare_for_extend / prepare_for_decode(req_to_token 写入 full loc)
  → ModelRunner.forward
      → HeadReallocAttnBackend.init_forward_metadata
          decode: _update_comp_mapping_decode → window indices(已翻译 comp 地址)
          extend: _update_comp_mapping_extend → comp prefix indices
      → 每层 RadixAttention.forward
          → set_kv_buffer(_fused_kv_write: full+comp 双写)
          → forward_decode / forward_extend(full Q 组 + comp Q 组分别 attention)
  → 采样 → ZMQ 回 TokenizerManager → detokenize → HTTP 响应
  → free(request finish):allocator.free(full loc 归还 + comp chunk 反推释放 + mapping 清零)
```

---

## 问题 1:head mask 在哪个时刻加载和 TP shard?

**加载时刻**:ModelRunner 初始化阶段(model 权重加载后、KV pool 分配前)。
`model_runner.py` L1552-1553(`enable_rlkv_inference` 分支):

```python
self.rlkv_head_masks = self._load_rlkv_head_masks()
sink = self.server_args.sink_window_size; recent = self.server_args.recent_window_size
```

**加载流程**(`_load_rlkv_head_masks`,L1737-1805):
1. 读 `adapter_weights.tsv`,不存在则 fallback `full_attention_heads.tsv`
2. `np.loadtxt` → `np.clip(0,1)` → **随机微扰** `np.random.uniform(0,1e-6)`(不可复现,D1)
3. `threshold = np.quantile(scores, sparsity)`;`mask = (scores >= threshold)`(sparsity 语义)
4. TP shard:`start = tp_rank * num_kv_heads_per_tp`,切片
5. 产物 `{layer_id: torch.Tensor[TP局部kv头]}`(float32, on device)

**backend 侧**:`HeadReallocAttnBackend.__init__`(L71)接收 mask,
`_precompute_head_indices`(L149)派生:
- `full_kv_head_indices / comp_kv_head_indices`(每层 KV head 分组)
- `full_q_head_indices / comp_q_head_indices`(KV head → Q head 展开,
  `kv_idx * num_kv_groups .. (kv_idx+1) * num_kv_groups`,L167-183)
- `restore_indices`(L188-192):`[full_q..., comp_q...]` → 原始 Q 序的映射

---

## 问题 2:Full/Compact tensor 的实际 shape?

`HeadReallocKVPool.__init__`(memory_pool.py L892-965),每层两个 buffer 对:

```text
full_k_buffer[l] / full_v_buffer[l] : [size_full + 1, n_full_l, head_dim]
comp_k_buffer[l] / comp_v_buffer[l] : [size_comp + 1, n_comp_l, head_dim]
```

- `size_full = Tf`(预算公式,model_runner.py L1571-1575)
- `size_comp = Tc = max_num_reqs × window`(L1569)
- `n_full_l / n_comp_l` = TP 局部第 l 层的 full/comp KV head 数(随层变化)
- +1 为 dummy slot(位置 0)
- 存储 dtype 与模型一致(bf16);`store_dtype != dtype` 时 view 转换(L979-999)

---

## 问题 3:full_to_comp_mapping 何时写入?

| 时刻 | 位置 | 说明 |
| --- | --- | --- |
| 分配 | allocator.py L329-335 | `zeros(size_full+2, int64)` on device,引用挂到 pool |
| decode 写入 | backend L217-264 | `_update_comp_mapping_decode`,init_forward_metadata(L360)调用 |
| extend 写入 | backend L266-320 | `_update_comp_mapping_extend`,init_forward_metadata(L443)调用 |
| 释放清零 | allocator.py L382 | `full_to_comp_mapping[free_index] = 0` |
| CUDA Graph | backend L575/L640 | capture/replay 均先调 `_update_comp_mapping_decode` |

写入索引语义:decode 用 `out_cache_loc`(req_to_token[req_idx, pos]);extend 用
`full_locs`(req_to_token[req_idx, prefix_len:seq_len])。两者都是 full pool 地址,
mapping 是"full loc → comp loc"的逐 token 翻译表。

---

## 问题 4:每个 request 的 compact chunk 何时分配?

`_get_comp_base`(backend L194-215),首次 extend 时分配:

```python
first_full_loc = self.req_to_token[req_pool_idx, 0]   # position 0 的 full loc
if first_full_loc > 0:
    existing = allocator.full_to_comp_mapping[first_full_loc]
    if existing > 0:
        return (existing - 1) // window * window + 1   # 反推已有 base
return allocator.alloc_comp_window()                   # 新分配
```

- **首次**:extend 路径 `_update_comp_mapping_extend`(L286)调 `_get_comp_base`
  → `alloc_comp_window()` 弹一个 chunk,返回 `comp_base = 1 + chunk_id*V`
- **后续**:decode(L240-250)与后续 extend 均通过 position-0 mapping 反推,不新分配
- chunk 隔离:每个 request 独占一个连续 V 槽 chunk,无跨 request 串扰

---

## 问题 5:sink slot 和 recent ring slot 如何计算?

统一公式(backend L252-258 decode / L309-314 extend):

```text
is_sink    = pos < sink
comp_off   = is_sink ? pos : sink + (pos - sink) % recent
comp_index = comp_base + comp_off
```

extend 的窗口判定(L301-307)额外限制 recent 区:

```text
recent_start = max(sink, seq_len - recent)
is_recent    = pos >= recent_start
in_window    = is_sink | is_recent     # 窗口外 token mapping = 0
```

即:extend 只给"最终 sink ∪ 最终 recent"写 mapping,窗口外 token 的 comp
mapping 保持 0(写 dummy,见问题 10)。

---

## 问题 6:extend 如何构造 compressed prefix?

`init_forward_metadata` extend 分支(L439-514):
1. `_update_comp_mapping_extend`(L443)先更新 mapping
2. full heads:`kv_indptr = cumsum(prefix_lens)`,`kv_indices` 用
   `create_flashinfer_kv_indices_triton` 构造(L446-464)—— 全 prefix
3. comp heads(L468-499):
   - `window_prefix_lens = min(prefix_len, V)`;`comp_kv_indptr = cumsum(...)`
   - `comp_kv_indices_raw` 用 `_build_sink_recent_indices`(L487,基于 req_to_token 的
     **full loc**)
   - `comp_kv_indices = kv_pool.translate_loc_full_to_comp(comp_kv_indices_raw)`
     (L497)—— 翻译为 comp pool 地址
4. `forward_extend` 的 comp Q 组(L906-908)用 `comp_kv_indices/comp_kv_indptr`
   替代 full 的 kv_indices/kv_indptr

`_build_sink_recent_indices`(L938+)Triton kernel:seq_len <= V 时全拷贝;
否则 sink `[0,sink)` + recent `[seq_len-recent, seq_len)` 两段。

---

## 问题 7:decode 如何覆盖 recent ring?

`_update_comp_mapping_decode`(L217-264):
1. 新 token 位置 `pos = seq_len - 1`(seq_lens 已递增)
2. 取 `out_cache_loc = req_to_token[req_idx, pos]`
3. 反推 `comp_base`(position-0 mapping,L241-250)
4. `comp_off = pos < sink ? pos : sink + (pos - sink) % recent`(L252-258)
   → **环形**:pos 递增时 comp 槽在 recent 区内循环覆盖
5. `full_to_comp_mapping[out_cache_loc] = comp_base + comp_off`(L264)

`init_forward_metadata` decode 分支(L358-415):
- full:`kv_indptr/kv_indices`(全历史,L366-378)
- comp:`window_kv_lens = min(seq_len, V)`;`window_kv_indices` 用
  `_build_sink_recent_comp_indices`(L394,带 mapping 翻译,直接产出 comp 地址)
- `forward_decode` comp Q 组(L805-818)用 `window_kv_indptr/window_kv_indices`

---

## 问题 8:request finish / free-group 如何释放 compact chunk?

`HeadReallocAllocator.free`(allocator.py L367-384):

```python
def free(self, free_index):
    if self.is_not_in_free_group:
        self.full_allocator.free(free_index)                 # 1. full 槽归还
        comp_indices = self.full_to_comp_mapping[free_index] # 2. 查 mapping
        non_zero = comp_indices[comp_indices > 0]
        comp_bases = (non_zero - 1) // window * window + 1   # 归一 chunk base
        for cb in torch.unique(comp_bases):                  # 防重复释放
            self.free_comp_window(cb)                        # 3. comp chunk 归还
        self.full_to_comp_mapping[free_index] = 0            # 4. mapping 清零
    else:
        self.free_group.append(free_index)                   # 延迟批释放
```

- free-group 模式:`is_not_in_free_group=False` 时 append,`flush` 时统一走上述逻辑
- 同一 chunk 多 loc 释放:`unique(comp_bases)` 保证只 free 一次

---

## 问题 9:CUDA Graph 额外保存哪些 metadata?

`init_cuda_graph_state`(L538-567)预分配:
```text
cuda_graph_kv_indices         : [max_num_tokens × max_context_len]   # full
cuda_graph_window_kv_indices  : [max_num_tokens × window_size]       # comp(已翻译)
cuda_graph_kv_indptr          : [max_bs + 1]
cuda_graph_window_kv_indptr   : [max_bs + 1]
cuda_graph_num_kv_splits      : [max_num_tokens]
cuda_graph_window_num_kv_splits: [max_num_tokens]
```

capture(L569)/replay(L634)逻辑:
- 每次先 `_update_comp_mapping_decode`(mapping 是**动态输入**,不在 graph 内)
- full/comp indices 用预分配 buffer + `create_flashinfer_kv_indices_triton` /
  `_build_sink_recent_comp_indices` 就地重建
- **本质**:window indices 本身在 graph 内(地址由 mapping 提供),mapping 每步
  更新后,同一份 indices buffer 指向不同 comp 槽 —— 这是 head-wise 双池
  CUDA Graph 的核心机制

---

## 问题 10:compact pool 耗尽时当前代码如何处理?

**静默降级,不报错**(计划书 Phase 5 要点 5 的现状):

```text
alloc_comp_window() → _comp_free_chunks 空 → return 0
_get_comp_base()    → return 0
extend: _update_comp_mapping_extend L287-288: if comp_base == 0: continue
        → 该 request 所有 comp mapping 保持 0 → comp heads attend dummy slot
decode: L259-263: comp_bases=0 → comp_indices=0 → mapping 写 0
        → comp heads attend 位置 0(dummy),数据错误
```

后果:超过 `max_comp_chunks` 的请求的 comp heads 静默读到 dummy slot 的 KV,
输出错误且无任何提示。**MVP 必须改 fail-fast**(设计文档 §4.5)。

---

## 附带发现(known_limitations 素材)

1. `_update_comp_mapping_extend` 是 **Python for 循环**(L280-320),bs 大时开销明显
2. extend 窗口外 token mapping=0 → `_fused_kv_write` 把它们写 comp dummy slot
   (注释自认 "harmless",L275),有无效写入开销
3. decode 反推 comp_base 依赖 position-0 的 mapping;若调度器 evict 了
   position 0,反推失效(当前无 eviction 路径,预留风险)
4. `_load_rlkv_head_masks` 的随机微扰(L1764)导致 mask 不可复现
5. extend 的 recent 判定 `max(sink, seq_len - recent)` 在 chunked extend 下
   的语义需单测验证(与一次性 extend 的 comp 落盘一致性)
