# Phase 4 阶段报告:Attention 语义正确性

> 日期:2026-08-13 | 状态:**完成(Gate 4 通过)**

## 目标(计划书 §6 Phase 4)

1. Tensor-level reference(纯 PyTorch)✓
2. Official DuoAttention E2E oracle ✓(Phase 0 单请求 + 本阶段协议)
3. Gate 4:tensor reference 全过 + E2E 无系统性分叉 ✓

## 4.1 Tensor-level reference

`benchmarks/headkv/ref_attention.py`:纯 torch 最小实现
- full head:标准 causal attention
- comp head:sink ∪ recent 窗口 mask(含 GQA 展开)

`test/srt/headkv/test_headkv_attention.py`(45 单测全绿,CPU):
- extend mapping:长序列仅 sink+recent 映射;短序列全映射;槽位公式
- decode 环形覆盖:相隔 recent 的 token 映射同一槽(覆盖语义)
- comp_base 复用:同一 request 多次 extend 共享 chunk
- chunked extend:recent 窗口相对最终 seq_len 滚动
- ref 窗口数学与 mapping 槽位集合一致

## 4.2 E2E correctness(2026-08-13 23:00)

**关键方法学发现:FullKV baseline 必须用 triton backend**
- FullKV 默认 flashinfer,DuoKV 强制 triton → 逐 token 分叉(非语义问题)
- 对照实验:FullKV-triton vs DuoKV(ratio=1.0,全 full)= **11/11 逐 token 一致**
  → 证明 HeadKV 接入零语义漂移,kernel 差异是唯一分叉源
- 实验协议更新:FullKV baseline 统一 `--attention-backend triton`

**对齐协议后结果(FullKV-triton vs DuoKV ratio=0.5):**
```text
20 条 prompts:20/20 逐 token 完全一致(1.00)
   - 覆盖 short_qa / medium / math / code / retrieval / long_instruction
4K 长 prompt:逐 token 一致(filler 场景)
首 token 一致率:20/20 = 100%(Gate 4 要求 ≥95%)
Mini-NIAH(4K, depth 0.2/0.5/0.8, 3 seeds):9/9 无损(FullKV 与 DuoKV 均 1.00)
```

## Gate 4 判定

- [x] Tensor reference 全部通过(45 单测)
- [x] E2E 无系统性分叉(20/20 逐 token 一致;首 token 100%)
- [x] 短于 window 的 prompt:DuoKV == FullKV(逐 token)
- [x] 大面积分叉定位:kernel 差异(ratio=1.0 对照实锤),非 comp bug

## 面试素材

- "接入零语义漂移":ratio=1.0 对照实验证明 HeadKV 双池在无压缩时与
  FullKV-triton 逐 token 一致 —— 干净归因,排除 kernel 混淆
- 20/20 逐 token 一致(短 prompt)+ 9/9 NIAH 无损(4K):可追问的具体数字
- 方法论:实验协议必须同 kernel 对比(flashinfer vs triton 差异实测污染)

## 下一步

- Phase 5:物理双池与容量 Gate(pool 单测 + Tf>T0 验证 + byte accounting)
