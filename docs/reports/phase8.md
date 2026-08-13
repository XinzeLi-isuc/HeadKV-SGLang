# Phase 8 阶段报告:项目整理与交付

> 日期:2026-08-13 | 状态:**完成(MVP 全部阶段 Gate 0-7 通过)**

## 交付物清单(计划书 §8 对照)

| 要求 | 位置 | 状态 |
| --- | --- | --- |
| README.md(复用/新增/修改/验证/未支持) | README.md | ✓ |
| docs/headkv/env.md | 环境冻结 + 运行入口 | ✓ |
| docs/headkv/architecture.md | 三者关系与生命周期 | ✓ |
| docs/headkv/rlkv_callgraph.md | 10 问逆向 | ✓ |
| docs/headkv/known_limitations.md | K1-K9(1 已修) | ✓ |
| artifacts/*.log | smoke/gate3/正确性 | ✓ |
| results/*.csv | capacity/throughput/online/quality | ✓ |
| figures/*.png | 容量 2 图 | ✓ |
| docs/reports/phase0-7.md | 每阶段报告 | ✓ |
| docs/interview/ | pitfalls + narrative | ✓ |

## 面试素材沉淀

- pitfalls.md:P1 代理劫持(方法论)/ P2 ninja PATH / P3 transformers 5 连坑 /
  P4 accelerate / P5 CUDA_VISIBLE_DEVICES
- narrative.md:一句话/30s 故事/关键数字表/STAR/深挖准备/边界声明

## 自检(对照计划书 §3.2 MVP 必须完成)

- [x] 单 GPU、TP=1
- [x] Llama-3.1-8B(GQA)decoder-only
- [x] DuoAttention 官方 pattern 加载、确定性二值化、维度校验
- [x] KV-head 级 Full/Compact partition
- [x] 复用 RLKV HeadReallocKVPool/Allocator/Backend
- [x] Compact Pool 只存 sink+recent,无完整历史副本
- [x] eager extend/decode/continuous batching
- [x] allocator 释放、环形覆盖、request interleave 正确
- [x] FullKV vs DuoKV 真实 token capacity 对比(1.88x)
- [x] NIAH + LongBench 2 子任务
- [x] 同一 SGLang runtime 下的吞吐/并发实验

## 成功等级(计划书 §10)

**A 级达成**(秋招主项目):
- policy 从 runtime 解耦 ✓ / 真实物理双池 ✓ / max_total_tokens 提升 ✓ /
  continuous batching 无泄漏 ✓ / 质量可接受 ✓ / 完整实验 ✓

S 级升级路线(未做,记录):
- 同一 runtime 支持 RLKVPolicy 对照 / current-main 迁移 / CUDA Graph 开启

## 项目名

```text
DuoKV-SGLang:基于 Head Reallocation 的异构 KV Cache Serving
```

## 遗留与建议

1. CUDA Graph 开启后可缓解单请求 -15~24% 开销(候选深化)
2. current-main 迁移需重新定位 4 个接入点(设计文档 §7 已给框架)
3. LongBench 官方评估模板跑全量(当前简化 F1)
4. git 状态:fork 分支 feat/headkv-duo(6 commits),项目仓库(12 commits)
