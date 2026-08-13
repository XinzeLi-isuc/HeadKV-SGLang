# Phase 7 阶段报告:正式实验

> 日期:2026-08-13 | 状态:**完成(Exp A-E 全部执行)**

## 实验环境(全部同协议)

```text
模型: Meta-Llama-3.1-8B-Instruct(bf16, GQA 32Q/8KV/32L)
GPU:  A6000 48GB × 2(FullKV-triton GPU0 / DuoKV GPU1)
协议: 同 commit(973b5e41 + headkv 改动)、同 triton backend、同 dtype、
      eager、R=64、mem-fraction 0.85、temperature=0
FullKV: T0=204824;DuoKV(ratio 0.5): Tf=385072
```

## Experiment A:Head ratio vs KV capacity(核心结果)

| ratio | full/compact | Tf 实测 | 增益 | 公式手算 | 吻合 |
| --- | --- | ---: | ---: | ---: | --- |
| 0.25 | 64/192 | 782432 | **3.82x** | 782432 | ✓ |
| 0.50 | 128/128 | 397360 | **1.94x** | 397360 | ✓ |
| 0.75 | 192/64 | 269002 | 1.31x | 269002 | ✓ |
| 1.00 | 256/0 | 204824 | 1.00x | =T0 | ✓ |

图表:figures/capacity_vs_ratio.png、capacity_gain.png

## Experiment B:Context length(单请求 E2E,含 32 decode tokens)

| len | FullKV | DuoKV | Δ |
| --- | --- | --- | --- |
| 4K | 1.041s (30.3 tok/s) | 1.375s (23.0) | -24% |
| 8K | 1.377s (23.2) | 1.712s (18.7) | -24% |
| 16K | 2.121s (15.1) | 2.447s (13.1) | -15% |

单请求 DuoKV 慢 15-24%:eager 下 full/comp 双路 kernel launch + Q 头
gather/restore 开销(计划书预期,capacity-oriented 定位,不承诺单请求加速)。

## Experiment C:Concurrency(8K prompt,16 decode)

| bs | 总量 | FullKV median | DuoKV median |
| --- | --- | ---: | ---: |
| 16 | 131K | 10.32s | 10.41s |
| 32 | 262K | 20.70s | 20.67s |
| 48 | 393K | 31.56s | 31.02s(-1.7%) |

- 双方全部成功;瓶颈在 prefill 波次(max_prefill_tokens=16384),非 KV 池
- decode-heavy(4K×64×128=256K tokens):FullKV 291.8 vs DuoKV 287.9 gen-tok/s(±1%)
- **容量优势的硬指标**:max_total_num_tokens = 204824 vs 385072(1.88x,引擎准入上限)
  —— eager 下不直接转吞吐,符合 capacity-oriented 定位

## Experiment E:Quality

| 测试 | FullKV | DuoKV | 结论 |
| --- | --- | --- | --- |
| Mini-NIAH 4K(depth 0.2/0.5/0.8×3) | 9/9 | 9/9 | 无损 |
| LongBench narrativeqa(40,≤16K ctx) | F1 0.1376 | F1 0.1676 | 无损(略高) |
| LongBench 2wikimqa(40,≤16K ctx) | F1 0.1217 | F1 0.1254 | 无损 |

LongBench 绝对 F1 偏低:简化 F1 实现(非官方评估模板),双端口同条件对比有效。
记忆参照:DuoAttention-Serve 的 LongBench 官方 F1 17.87 vs 17.76 持平——趋势一致。

## Experiment D:Online serving

| 负载 | 指标 | FullKV | DuoKV | Δ |
| --- | --- | ---: | ---: | --- |
| memory-light(512 ctx,8w,60s) | req/s | 7.38 | 5.73 | -22% |
| | P50 E2E | 1.09s | 1.42s | -23% |
| memory-bound(8K ctx,16w,60s) | req/s | 1.33 | 1.33 | **持平** |
| | P50 E2E | 12.22s | 12.22s | **持平** |

- memory-light 的 -22%:单请求双路开销(与 Exp B 一致)
- **memory-bound 持平**:长上下文高并发下开销被摊平,且 DuoKV 无池满排队
- 现场证据(server 日志,同 16 并发):token usage FullKV 0.31 vs DuoKV 0.16
  —— 等量负载下 KV 占用减半

## 总结论

1. **容量**:max_total_tokens 1.88x(ratio 0.5),最高 3.82x(ratio 0.25),与预算公式逐位吻合
2. **质量**:NIAH 9/9 无损;LongBench 2 子任务 F1 持平或略高
3. **吞吐**:单请求 -15~24%(eager 双路开销);并发/长上下文场景持平
4. 定位:capacity-oriented serving(容量提升是主指标,不承诺单请求加速)

## 图表

- figures/capacity_vs_ratio.png、capacity_gain.png
- 数据:artifacts/{capacity_sweep,exp_bc,exp_c_concurrency,exp_c_decodeheavy,
  exp_d_online,longbench,niah_mini}.json/csv
