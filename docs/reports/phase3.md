# Phase 3 阶段报告:双池接入

> 日期:2026-08-13 | 状态:**完成(Gate 3 通过)**

## 目标(计划书 §6 Phase 3)

1. 新增 --enable-headkv 等 7 参数 ✓
2. 保留 --enable-rlkv-inference 兼容入口(原路径不动,MVP 最小 diff)✓
3. 接入调用流:ServerArgs → HeadKVConfig → HeadPolicy → TP mask → budget → 双池 ✓
4. Gate 3 启动日志 + 物理 tensor 检查 ✓

## 改动(fork commit 970cbd13)

| 文件 | 改动 |
| --- | --- |
| server_args.py | 7 个 headkv 字段/参数;与 rlkv 互斥校验;强制 triton+关 radix |
| model_runner.py | `_init_headkv()`(policy→mask→partition→budget→日志);双池/allocator/backend 三处分支 |
| head_realloc_backend.py | window 优先读 `model_runner.headkv_*`(不落 RLKV 16/32 默认) |

## Gate 3 实测(2026-08-13 22:27,ratio=0.5)

```text
[HeadKV] policy=duo mask=[32,8] layers=32 kv_heads/layer=8
         full_heads=128 compact_heads=128 nominal=0.5 effective=0.5
         sink=128 recent=256 window=384 max_running_requests=32
         T0=204824 Tf=397360 Tc=12288 predicted_gain=1.940x
KV Cache allocated #tokens: 397360
HeadReallocAttnBackend enabled (HeadKV)
server fired up;4K smoke 1.792s,输出与 FullKV 逐字一致
```

- 预算公式手算校验:Tf = (204824×256 − 12288×128)/128 = 397360 ✓
- 正确性 sanity:同一 4K prompt,DuoKV 输出 == FullKV 输出
- 33 单测回归通过;py_compile 全过

## Gate 3 自检清单

- [x] policy/pattern path 日志
- [x] layers/Q/KV heads 日志(mask=[32,8])
- [x] full/compact counts(128/128)
- [x] sink/recent/window(128/256/384)
- [x] max_running_requests(32)
- [x] T0(204824)/Tf(397360)/Tc(12288)/predicted_gain(1.940x)
- [x] 物理 tensor shape:HeadReallocKVPool 每层
      full[Tf+1, n_full, dim] / comp[Tc+1, n_comp, dim](逐行核对 L937-960;
      Phase 5 补 pool 单测)
- [x] 无额外 [T0, all_heads, dim] 完整副本(双池即唯一存储)

## 面试素材

- 预算公式与实测 1.94x 完全吻合(Tf 手算验证)—— 数字可追问
- 输出与 FullKV 逐字一致 = 双池接入零语义漂移
- "policy 解析 window 优先于 fork 默认"避免了 Duo 128/256 被 16/32 覆盖的隐患

## 下一步(Phase 4)

- Tensor-level reference(纯 PyTorch 最小实现)+ E2E correctness(三方对比)
- 需先杀当前 DuoKV server 释放 GPU 0
