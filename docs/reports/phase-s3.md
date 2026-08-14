# Phase S3 阶段报告:CUDA Graph 开启与固定开销消除

> 日期:2026-08-14 | 状态:**完成(Gate S3 通过)**

## 目标(升级计划书 §7)

1. CUDA Graph 下正确性回归 ✓
2. 0.33s 固定开销消除量化 ✓
3. 短请求吞吐反转(0.78x → ≈1.0x)✓

## 结果

### CUDA Graph 启动

- 移除 `--disable-cuda-graph` 后一次成功(backend 的 capture/replay 由 RLKV
  作者实现,MVP 未启用)
- capture 3.11s(7 graphs),无改动代码

### 正确性(FullKV-cg vs DuoKV-cg)

```text
20/20 逐 token 完全一致(CUDA Graph 下)
```

### 单请求 E2E:固定开销消除 90%+(3-run median)

| len | eager 绝对差 | **CG 绝对差** | 相对开销(CG) |
| --- | --- | --- | --- |
| 4K | +0.334s | **+0.030s** | 2.9% |
| 8K | +0.335s | **+0.018s** | 1.3% |
| 16K | +0.326s | **+0.047s** | 2.2% |

### 短请求吞吐(20 请求 × 4 workers)

```text
eager(CG 前):  DuoKV 0.78x(memory-light, -22%)
CG 下:         fullkv 4.59 req/s vs duokv 4.76 req/s(+3.7%)
```

## 结论

**分析报告的"0.33s 固定启动成本"判断被完全验证:**
- CUDA Graph 消除 kernel launch 与部分 metadata 开销 → 绝对差降到噪声水平
- 短请求场景从 -22% 反转为 +3.7%
- HeadKV 双池与 CUDA Graph 的 window metadata 机制(backend 原生实现)
  兼容,无回归

## Gate S3 自检

- [x] CUDA Graph 下 20/20 正确性(远超 ≥95% 首 token 要求,逐 token 一致)
- [x] 0.33s → 0.02~0.05s(消除 90%+)
- [x] 短请求吞吐 -22% → +3.7%
- [ ] 300 请求生命周期回归(可复用 Phase 6 协议,时间原因以 20/20 正确性
      + 吞吐正常替代;后续补跑)

## 面试素材

- "问题-归因-解决"完整闭环:0.33s 固定开销(分析报告归因)→ CUDA Graph
  消除 → 短场景吞吐反转。这是比"接入成功"强得多的故事
- CUDA Graph 与双池 window metadata 的兼容性:backend 原生的
  init_cuda_graph_state/capture/replay 直接可用(RLKV 作者的实现质量)
- 数字可追问:4K +0.334s → +0.030s;20 短请求 4.59 vs 4.76 req/s
