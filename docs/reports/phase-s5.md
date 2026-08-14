# Phase S5 阶段报告:文档交付与 S 级收尾

> 日期:2026-08-14 | 状态:**完成(S 级达成)**

## 交付物(升级计划书 §9 对照)

| 要求 | 位置 | 状态 |
| --- | --- | --- |
| README 更新(S 级状态/双 policy/CUDA Graph) | README.md | ✓ |
| 实验复现报告(可完美复现) | EXPERIMENTS.md | ✓ |
| 阶段报告 S0/S1/S2/S3/S5 | docs/reports/phase-s*.md | ✓ |
| 面试叙事更新(双 policy + CUDA Graph 闭环) | docs/interview/narrative.md | ✓ |
| 踩坑 P6 追加 | docs/interview/pitfalls.md | ✓ |
| 图表 | figures/s_head_distribution.png 等 | ✓ |
| git 逐 commit | fork 分支 + 项目仓库 | ✓ |

## S 级判定(升级计划书 §2.2)

- [x] 同一 runtime 同时支持 DuoAttentionPolicy 与 RLKVPolicy
      (HeadPolicy.create 工厂统一分发;老 RLKV 入口收敛;双入口输出逐字一致)
- [x] 双 policy 容量/质量/吞吐对照实验完成
      (容量由 F/C/V 决定;NIAH 双 9/9;LongBench Duo 略优;4K E2E ±10%)
- [x] RLKVPolicy 与原 loader 输出一致(确定性;差异仅来自随机微扰)
- [x] head 分布差异分析(Jaccard=0.446;Duo 层内均匀 vs RLKV 层间自由)
- [x] (增强)CUDA Graph 开启:0.33s → 0.03s,短请求 -22% → +3.7%

## S 级成果汇总

```text
容量:     1.94x(0.5)/ 3.82x(0.25),公式逐位吻合
质量:     NIAH 9/9;LongBench F1 持平或略高(双 policy)
正确性:   20/20 逐 token(eager + CUDA Graph)
生命周期: 1500+300 请求零失败
开销:     0.33s 固定 → 0.03s(CUDA Graph 消除 90%+)
吞吐:     短请求 -22% → +3.7%;decode 94~165 tok/s
双 policy: 同一 runtime,输出一致,Jaccard=0.446
```

## 面试叙事升级(30s)

DuoAttention 与 RLKV 用完全不同的方法识别关键 KV heads(score 阈值 vs
训练 adapter),但部署都归约为 Full/Compact 两种生命周期。项目把 RLKV
算法专属的 mask loader 与 runtime 解耦,抽象出 policy-decoupled 后端:
同一套双池 runtime 可装载任意 head 选择策略,完成两种算法的容量/质量/
分布对照;并归因-消除单请求固定开销(0.33s → CUDA Graph → 0.03s)。

## 遗留(投递后深化)

1. Phase S4:current-main 迁移(两天止损框架已备)
2. LongBench 官方模板全量(当前简化 F1,双 policy 对照有效)
3. NIAH 8K/16K 完整版
4. CUDA Graph 300 请求回归已补(CG 下 300/300)
