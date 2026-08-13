# HeadKV 架构文档(architecture.md)

> Phase 1 交付物,2026-08-13。核心:Full loc、Compact loc、request slot
> 三者关系与生命周期。基于 RLKV fork @ 973b5e41。

## 1. 地址空间总览

```text
Full pool(单实例):  slot 0 = dummy, 1..Tf 由 full_allocator 管理
  [0][1][2][3][4][5][6][7]...[Tf]
   ^  dummy(不可用)

Comp pool(单实例):  slot 0 = dummy, 1..Tc;按 request 划分 chunk
  [0] |--- chunk 0 ---| |--- chunk 1 ---| ... |--- chunk R-1 ---|
       c_base=1          c_base=1+V
  chunk 内部布局(chunk 起始 = comp_base):
  [comp_base .. comp_base+sink-1]   = sink 区(固定,extend 时写)
  [comp_base+sink .. comp_base+V-1] = recent 环(环形覆盖)

req_to_token_pool(单实例): req_pool_idx → full loc 序列(行式)
  req 0: [fl_0][fl_1][fl_2]...
  req 1: [fl_0'][fl_1']...
```

## 2. 三者关系(核心不变量)

```text
req_to_token[req_pool_idx, pos] = full_loc        # 逻辑位置 → full 地址
full_to_comp_mapping[full_loc]  = comp_loc(或 0)  # full 地址 → comp 地址
comp_loc ∈ [comp_base(req), comp_base(req)+V)     # 每 request 单 chunk
```

- 一个 request 的所有 token 共享**一个** comp chunk(由 position-0 mapping 反推)
- 同一 full_loc 永不重复映射(每 token 唯一 full loc → 唯一 comp 槽;
  recent 环覆盖的是**不同** full_loc 指向同一 comp 槽,即"新 token 顶掉旧 token")
- free 的唯一依据 = mapping(full loc 序列 → 反推 chunk base)

## 3. 生命周期时序(新 request → 释放)

```text
t0  enter: 调度器准入(available_size 检查,不占 comp)
t1  extend: full_allocator.alloc(extend_len) → full locs(连续)
            _get_comp_base → alloc_comp_window() → comp_base(首次)
            set_kv_buffer: _fused_kv_write 双写(full 全写;comp 仅 in_window)
            _update_comp_mapping_extend: 窗口内 full_loc → comp_loc 映射
t2  decode: full_allocator.alloc(1) → 新 full_loc
            set_kv_buffer: 写 full 新槽 + comp 环形槽
            _update_comp_mapping_decode: out_cache_loc → comp 环形槽
            (seq < sink 时写 sink 区;否则写 recent 环)
t3  finish: free(full_locs):
            1) full_allocator.free
            2) mapping 反推 comp_base → free_comp_window(chunk 回池)
            3) mapping[full_locs] = 0
t4  reuse: 新 request 可复用同一 full loc 与同一 chunk id
```

## 4. 组件职责与依赖

| 组件 | 职责 | 关键接口 |
| --- | --- | --- |
| `HeadReallocKVPool` | 物理存储:每层 full/comp 双 buffer | `set_kv_buffer`(fused 双写)、`get_comp_*_buffer`、`translate_loc_full_to_comp` |
| `HeadReallocAllocator` | full 池 alloc/free + comp chunk 管理 | `alloc`、`free`、`alloc_comp_window`、`free_comp_window`、`available_size` |
| `HeadReallocAttnBackend` | 每步 mapping 维护 + full/comp 双路 attention | `init_forward_metadata`、`forward_decode`、`forward_extend` |
| `ModelRunner` | mask 加载、预算计算、双池与 backend 组装 | `_load_rlkv_head_masks`、双池 init(L1552-1597)、backend 选择(L1858-1867) |
| `ServerArgs` | 参数(RLKV 专属 + 通用) | `--enable-rlkv-inference`、`--rlkv-sparsity`、sink/recent |

## 5. 数据流时序(forward 单步)

```text
ModelRunner.forward
  └─ HeadReallocAttnBackend.init_forward_metadata(batch)
       ├─ decode: _update_comp_mapping_decode(先更新 mapping)
       │          full:  create_flashinfer_kv_indices_triton(全历史)
       │          comp:  _build_sink_recent_comp_indices(mapping 翻译)
       └─ extend: _update_comp_mapping_extend(先更新 mapping)
                  full:  kv_indices(全 prefix)
                  comp:  _build_sink_recent_indices + translate_loc_full_to_comp
  └─ 每层 RadixAttention
       ├─ set_kv_buffer → _fused_kv_write(full+comp 分流)
       └─ backend.forward_decode / forward_extend
            ├─ full Q 组: decode_attention_fwd(kv_indptr/kv_indices)  # 全历史
            └─ comp Q 组: decode_attention_fwd(window_kv_indptr/window_kv_indices)
            → restore_indices 还原 Q 序 → o_full
```

## 6. 扩展点(HeadKV 接入位置,见 DESIGN.md §2/§5)

- `ServerArgs` → `HeadKVConfig`(policy 参数)
- `ModelRunner._load_rlkv_head_masks` → `HeadPolicy.load_global_kv_mask`
- `HeadReallocKVPool/Allocator` → 复用(修边界 bug)
- `HeadReallocAttnBackend` → 复用(通用化命名)
