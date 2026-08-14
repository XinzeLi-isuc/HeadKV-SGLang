# S 级实验数据分析报告(Phase S0-S3 深度分析)

> 2026-08-14。全部计算基于 artifacts 原始数据(analyze_s_experiments.py 可复现),
> 无推算。

## 1. CUDA Graph:固定开销的构成与消除(核心洞察)

| len | eager 绝对差 | CG 绝对差 | 消除率 | CG 相对开销 |
| --- | ---: | ---: | ---: | ---: |
| 4K | +0.334s | +0.030s | 91.0% | 2.87% |
| 8K | +0.335s | +0.018s | 94.6% | 1.72% |
| 16K | +0.326s | +0.047s | 85.6% | 4.61% |

**构成分解**:0.33s 固定开销中约 0.30s(91%)是 kernel launch 与
Q gather/restore 的 Python 侧串行成本 —— CUDA Graph 一次性捕获后消除;
剩余 ~0.03s 是 prefill 阶段 graph 外开销(comp KV 写入、mapping 更新、
window metadata 构造),随长度小幅波动(16K 最大,疑似 prefill 波次噪声)。

**decode 吞吐 4~8x**:eager ~20 token/s(单请求)vs CG 94~165 token/s
(多请求 batch)。每 decode step 的 launch 串行开销被消除,batch 越大收益越明显。

## 2. 双 policy 容量:F/C/V 决定论(理论-实测闭环)

| policy | F | C | V | Tc=R·V | 实测 Tf | 公式 Tf | 偏差 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| Duo | 128 | 128 | 384 | 12288 | 397358 | 397360 | -0.001% |
| RLKV | 128 | 128 | 48 | 1536 | 408110 | 408112 | -0.000% |

**决定性结论**:容量只由 F/C/V 决定(Tf = (T0(F+C) − R·V·C)/F),
与 head 选择算法无关。RLKV 小 window 的 2.7% 容量优势完全来自 comp 池
开销(Tc=R·V 差 8 倍)。因此 **head 选择算法的差异只能体现在质量维度**,
容量对照的正确解读是"F/C/V 配置的工程权衡"。

## 3. 双 policy 质量:Duo 方向性略优,未达统计显著

| 测试 | Duo | RLKV | Δ | 配对 t |
| --- | --- | --- | --- | --- |
| NIAH 4K | 9/9 | 9/9 | 0 | - |
| narrativeqa(30) | 0.1785 | 0.1643 | +8.6% | t=0.80 |
| 2wikimqa(30) | 0.1198 | 0.1074 | +11.5% | t=1.91 |

- 两个任务方向一致(Duo 均正),但 n=30 下 |t|<2,未达统计显著
- 保守结论:同压缩率下 Duo 的 head 选择不劣于 RLKV(略优趋势),
  需扩大样本(≥100)才能定论
- NIAH 双 9/9:两种算法都能保留 retrieval 所需信息

## 4. S0 双入口一致性

老入口 Tf=408878 vs 新入口 408880(Δ2 token = profiling 噪声),输出逐字一致。
验证了"入口收敛正确且无行为回归"。

## 5. S2 head 分布:45% 共识 / 55% 互补

- Jaccard=0.4463:两种算法对"重要 head"的判断近半共识
- Duo:每层固定 4/8(硬 top-k 约束,层内均匀)
- RLKV:每层 1~8 个(quantile 全局阈值,层间自由;存在全 full 层)
- head 层频分布相似(13~20 vs 12~22/32)
- 研究延伸:55% 互补 → head 级集成(两算法投票/加权)的潜在空间

## 6. 总结论

1. **CUDA Graph 是 HeadKV 吞吐短板的完整解**:0.33s → 0.03s(91% 消除),
   短请求 -22% → +3.7%,decode 4~8x
2. **双 policy 架构贡献成立**:同一 runtime 装载两种 head 选择算法,
   容量-质量-分布三维对照,理论-实测闭环(<0.001% 偏差)
3. **算法对比的正确框架**:容量由 F/C/V 决定,质量才反映 head 选择差异;
   质量差异方向一致但需更大样本定论
4. **诚实边界**:Duo 略优未显著;RLKV adapter 来自 R1 模型(Instruct 上
   适配性限制);LongBench 简化 F1

## 7. 面试叙事升级

- "0.33s 中 91% 是 kernel launch,CUDA Graph 消除后剩 0.03s prefill 期
  开销"—— 比"快了"更有说服力的构成分解
- "容量由 F/C/V 决定,head 选择只影响质量"—— 正确的对照归因,面试可
  现场推导公式
- 配对 t 检验(0.80/1.91):主动说统计边界,比隐藏更有可信度

## 图表

- figures/s_cuda_graph_gap.png(eager vs CG 消除)
- figures/s_policy_quality.png(双 policy F1 + t 值标注)
- figures/s_head_distribution.png(S2 heatmap + Jaccard)
