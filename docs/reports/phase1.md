# Phase 1 阶段报告:RLKV 调用链逆向

> 日期:2026-08-13 | 状态:**完成(Gate 1 通过)**

## 目标(计划书 §6 Phase 1)

只追踪 mask loader → pool budget → HeadReallocKVPool → HeadReallocAllocator →
HeadReallocAttnBackend → request free 的完整调用链,回答 10 个问题。

## 交付物

| 文件 | 内容 |
| --- | --- |
| `docs/headkv/rlkv_callgraph.md` | 10 问逐条回答,全部带真实行号 |
| `docs/headkv/architecture.md` | Full/Compact loc 与 request slot 三者关系 + 生命周期时序 |
| `docs/headkv/known_limitations.md` | K1~K9 现状问题(对应计划书修订表) |

## 10 问核心答案摘要

1. mask 加载:ModelRunner init(L1552),TP slice(L1783-1797),backend 派生
   full/comp Q/KV indices + restore(L149-193)
2. tensor shape:每层 `full: [Tf+1, n_full, dim]` / `comp: [Tc+1, n_comp, dim]`
   (memory_pool.py L937-960)
3. mapping 写入:decode/extend 各自 init_forward_metadata 前置更新;free 清零
4. comp chunk 分配:首次 extend `_get_comp_base → alloc_comp_window`;
   后续经 position-0 mapping 反推,不重复分配
5. 槽位公式:comp_base + (is_sink ? pos : sink + (pos-sink)%recent)
6. extend compressed prefix:`_build_sink_recent_indices` + translate_loc_full_to_comp
7. decode 环形覆盖:pos 递增 → recent 环滚动覆盖
8. free:full 归还 + mapping 反推 chunk + unique 防重 + mapping 清零
9. CUDA Graph:window indices 预分配在 graph 内,mapping 为动态输入
10. **comp 耗尽:静默降级(K1)—— 必须 fail-fast 修复**

## Gate 1 判定

- [x] 新 request 进入 → 释放的完整生命周期可画出
      (architecture.md §3 时序 t0~t4)
- [x] Full loc、Compact loc、request slot 三者关系明确
      (architecture.md §2 不变量)
- [x] 调用链每环都有真实行号支撑,无推测

## 关键发现(影响后续阶段)

1. **comp 池耗尽静默降级**(K1)是 Phase 5 的 fail-fast 修复点
2. extend mapping 更新是 Python 循环(K4),decode 已向量化——性能不对称,
   候选优化点(面试素材)
3. chunked extend 的 comp 落盘语义需单测(K5),进 Phase 4 测试矩阵
4. 预算公式现状带 T0 下限(K6),Phase 3 统一为设计公式

## 下一步

- Phase 2:HeadPolicy / Duo loader / partition / budget 实现(纯 CPU 单测)
