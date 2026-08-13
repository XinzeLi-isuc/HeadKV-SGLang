# Phase 5 阶段报告:物理双池与容量生死 Gate

> 日期:2026-08-13 | 状态:**完成(Gate 5 通过,项目生死线越过)**

## 目标(计划书 §6 Phase 5)

1. Compact heads 无完整历史 tensor ✓
2. 实际 pool bytes 与预算公式一致 ✓
3. compact chunk 数 == max_running_requests ✓
4. Full Pool 扩大转化为 max_total_tokens ✓
5. comp 池耗尽 fail-fast(修复 K1)✓

## 必做测试(test_headkv_pool.py,11 用例,CPU)

| 测试 | 结果 |
| --- | --- |
| test_pool_tensor_shapes | ✓ full[Tf+1,n_full,dim] / comp[Tc+1,n_comp,dim] |
| test_pool_tensor_shapes_all_comp | ✓ 全 comp 时 full buffer 0 head |
| test_no_full_history_copy | ✓ 无 [T0, all_heads, dim] 副本 |
| test_pool_byte_accounting | ✓ bytes == Σ nelement×element_size |
| test_comp_chunk_count | ✓ max_comp_chunks = Tc // V = 4 |
| test_comp_window_alloc_free_cycle | ✓ 分配/释放/复用 |
| test_free_derives_comp_base_once | ✓ 同 chunk 多 loc 只 free 一次 |
| test_free_group_flush | ✓ free_group_begin/end 批释放 |
| test_all_full_degenerate_path | ✓ |
| test_allocator_restores_initial_state | ✓ 100 次循环恢复初始态 |
| **test_comp_pool_exhaustion_fails_fast** | ✓ 修复后抛 RuntimeError |

## K1 修复(comp 池耗尽 fail-fast)

原实现:`_update_comp_mapping_extend` 在 `comp_base==0` 时静默 `continue`
→ comp heads attend dummy slot,错误输出无提示。
修复:抛 RuntimeError(含 req_pool_idx + 池状态 + 修复建议)。
改动:`head_realloc_backend.py` L287-299。**56 单测全绿**。

## Gate 5 容量实测(2026-08-13,GPU 0,mem-fraction 0.85)

```text
ratio   full/compact   Tf(实测)    Tf/T0    Tf(预算公式手算)   吻合
0.25    64/192         782432      3.82x   782432            ✓
0.50    128/128        397360      1.94x   397360            ✓
0.75    192/64         269002      1.31x   269002(floor)     ✓
1.00    256/0          204824      1.00x   = T0              ✓
```

- 4 个实测点严格单调递减(compact head 越多,容量越大)
- 公式例:0.25 → Tf = (204824×256 − 12288×192)/64 = 782432
- 数据:artifacts/capacity_sweep.csv + capacity_sweep_gate3.log

## Gate 5 判定(项目生死线)

- [x] 物理 KV bytes 下降 / Full Pool 扩大(Tf 3.82x at ratio 0.25)
- [x] Tf > T0(存在 compact heads 时,全部实测点)
- [x] 容量趋势与 full-head ratio 单调一致(4 点实测)

**项目成立:不是"只改 attention 计算但保存完整 KV",而是真实物理双池容量提升。**

## 面试素材

- 容量增益数字可追问:3.82x(0.25)/ 1.94x(0.5)/ 1.31x(0.75),且与预算公式
  手算逐位吻合 —— 理论-实测闭环
- K1 修复的工程价值:静默数据错误 → fail-fast,这是 serving 系统的
  可靠性设计(面试常问"生产系统如何防静默失败")

## 下一步

- Phase 6:request 生命周期与 continuous batching(1000 混合请求 invariant)
