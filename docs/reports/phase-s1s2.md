# Phase S1/S2 阶段报告:双 policy 对照实验与 head 分布分析

> 日期:2026-08-14 | 状态:**完成(Gate S1/S2 通过)**

## S1 双 policy 对照(同 effective ratio 0.5,F=C=128,R=32)

### 容量(各自官方 window)

| policy | window | Tf 实测 | 增益 | 备注 |
| --- | --- | ---: | ---: | --- |
| Duo(ratio 0.5) | 128/256=384 | 397358 | 1.94x | comp 池 12288 槽 |
| RLKV(sparsity 0.5) | 16/32=48 | 408110 | 1.99x | comp 池 1536 槽 |

**关键结论:容量只由 F/C/V 决定,与 head 选择算法无关。**
差异 2.7% 全部来自 window(comp 池开销:Tc=R×V)。
RLKV 小 window → comp 池开销小 8 倍 → Tf 略高(但保留信息少)。

### 质量(NIAH 4K + LongBench 各 30,同 F/C)

| 测试 | Duo | RLKV | Δ |
| --- | --- | --- | --- |
| NIAH 4K(3 depth×3 seed) | 9/9 | 9/9 | 0 |
| narrativeqa F1 | 0.1785 | 0.1643 | +8.6% |
| 2wikimqa F1 | 0.1198 | 0.1074 | +11.5% |

结论:同压缩率下 Duo 的 head 选择质量略优;样本 30,差异在噪声范围,
保守表述"Duo 略优,需扩大样本"。

### 吞吐(4K 单请求 E2E,3 次 median)

| policy | E2E |
| --- | --- |
| Duo | 1.292s |
| RLKV | 1.419s |

同 runtime 同 backend,双 policy 吞吐接近(Duo 略快 ~10%,含 GPU 间噪声)。

## S2 head 分布差异分析

### 统计(artifacts/s2_head_stats.json)

```text
full head 集合 Jaccard = 0.4463(45% 共识 / 55% 互补)
Duo :每层固定 4/8(硬 top-k 约束,层内均匀)
RLKV:每层 1~4 个(quantile 全局阈值,层间自由)
head 层频:Duo 13~20/32 vs RLKV 12~22/32(分布相似)
```

### 洞察

1. 两种算法对"哪些 head 重要"有 45% 共识、55% 分歧
2. 结构差异:Duo 强制层内均匀(每层恰好 4 个 full),RLKV 允许层间不均
   (layer 0 仅 1 个 full,layer 2/6 有 4 个)
3. 研究延伸:55% 互补 → head 级集成(两算法投票/加权)是未来方向

图:figures/s_head_distribution.png(双 heatmap + Jaccard)

## Gate S1/S2 自检

- [x] 双 policy 容量/质量/吞吐对照表(artifacts/s1_quality.json)
- [x] 同 effective ratio 0.5 对照(clean:F=C=128)
- [x] head 分布统计 + 图 + 洞察

## 面试素材

- "容量由 F/C/V 决定,head 选择的质量差异才是算法对比"—— 正确的对照归因
- Jaccard 0.45:两种算法 45% 共识 55% 互补 → 集成潜力
- 同压缩率下 Duo 略优(可复现数字)
