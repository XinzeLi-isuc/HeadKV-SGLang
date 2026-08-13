# Phase 2 阶段报告:HeadPolicy 抽象与 Duo loader

> 日期:2026-08-13 | 状态:**完成(Gate 2 通过)**

## 目标(计划书 §6 Phase 2)

1. HeadKVConfig ✓
2. HeadPolicy 抽象 ✓
3. DuoAttentionPolicy ✓
4. RLKVPolicy(兼容封装)✓
5. 确定性 top-k / threshold 二值化 ✓
6. TP-local partition 接口(MVP TP=1)✓

## 交付物(fork 分支 feat/headkv-duo,commit 9d3ff5c1)

```text
python/sglang/srt/headkv/
├── config.py        HeadKVConfig:校验 + window 优先级解析
├── policy.py        HeadPolicy 抽象 + create 工厂 + 模型配置鸭子类型取值
├── duo_policy.py    Duo pattern loader + 确定性二值化 + GQA 校验 + summarize()
├── rlkv_policy.py   RLKV adapter 兼容(去随机微扰)
├── manual_policy.py 人工 mask(调试)
├── partition.py     to_tp_local(完备且不相交)
└── budget.py        Tf/Tc 预算(计划书公式 + 边界)
test/srt/headkv/
├── test_duo_policy.py   19 用例
├── test_partition.py     5 用例
├── test_budget.py        8 用例
└── verify_gate2.py       Gate 2 实测脚本
```

## 测试结果

```text
33 passed in 1.71s(纯 CPU,无 GPU)
py_compile 全过
```

覆盖:官方 pattern 加载(真实路径)、shape 校验、确定性(两次一致)、同分
tie-break、window 优先级(CLI > deploy > config > 报错)、GQA 双路径(KV 粒度
不二次 OR / Q 粒度 OR)、threshold 语义、预算公式与全部边界、TP=1/2 partition。

## 关键发现(影响 Experiment A 设计)

**官方 threshold=0.5 的 pattern effective full ratio = 0.914**
(256 KV heads 中 234 full / 22 compact,实测 verify_gate2.py):

- 官方默认压缩率很低(仅 9% compact)→ 容量收益有限
- 原因:Duo 官方二值化是 threshold=0.5(score >= 0.5 即 full),而 RLKV 是
  sparsity-quantile(强制保留 top-50%)—— 两种语义完全不同
- **结论**:Experiment A 必须用 `--headkv-full-head-ratio` 档位(25/50/75%)
  控制 effective ratio;threshold=0.5 仅作"官方原样"对照
- 面试素材:识别"官方默认配置压缩率低"这一事实,主动调整实验设计,
  比直接跑官方默认更有说服力

## Gate 2 判定

- [x] 多次运行 mask 完全一致(确定性)
- [x] Full + Compact 覆盖全部 KV heads 且无重叠
- [x] summarize() 输出 nominal/effective ratio(启动日志就绪)
- [x] 官方 GQA pattern 不被错误二次 OR 聚合

## 其他

- rlkv-eval 补装 pytest 9.1.1(测试工具,非运行时依赖,已记录 env.md)
- Phase 3 前置:ServerArgs 参数 + ModelRunner 接入(需 GPU smoke)

## 下一步

- Phase 3:ServerArgs(--enable-headkv 等)+ ModelRunner 双池接入
- Phase 4:attention 语义正确性(tensor reference)
