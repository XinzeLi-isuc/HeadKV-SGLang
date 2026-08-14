# Phase S0 阶段报告:双 policy 统一入口

> 日期:2026-08-14 | 状态:**完成(Gate S0 通过)**

## 目标(升级计划书 §4)

1. 老入口 `--enable-rlkv-inference` 收敛到 HeadPolicy(RLKVPolicy)✓
2. runtime 无 policy 分支(双 policy 同一套双池/allocator/backend)✓
3. RLKVPolicy 单测 + 双入口一致性 ✓

## 代码改动(fork commit 08cf2d22 + 后续 1 个 fix)

| 文件 | 改动 | 为什么 |
| --- | --- | --- |
| model_runner.py | `_init_headkv()` 统一推导 policy:老入口→"rlkv",新入口→headkv_policy;pattern_path 兼容 adapter_load_path;rlkv 场景 window 默认 16/32 | 双入口收敛到一个配置构造点,runtime 无分支 |
| model_runner.py | 老 RLKV 分支(L1565)替换为与 headkv 相同的 `_init_headkv` 双池路径 | 删除直连 loader + 手写预算(带 max 下限),统一计划书公式 |
| model_runner.py | allocator/backend 分支合并(`enable_headkv or enable_rlkv_inference`),window 统一用 policy 解析值 | 消除 rlkv 分支读 server_args 默认 16/32 的隐式依赖 |
| headkv/config.py | HeadKVConfig 加 `sparsity` 字段 + [0,1] 校验 | RLKVPolicy 参数透传 |
| headkv/rlkv_policy.py | sparsity 从 cfg 读(默认 0.5);补 `summarize()` | 工厂统一 `factory(cfg)` 签名;Gate 3 日志依赖 |

## 踩坑 P6:新入口 rlkv 的 window 默认未生效

- 现象:`--enable-headkv --headkv-policy rlkv` 启动崩
  `TypeError: int() argument must be ..., not 'NoneType'`(sink=None)
- 根因:window 默认 16/32 只挂在 `is_rlkv_entry`(enable_rlkv_inference),
  新入口显式 policy="rlkv" 时 headkv_sink_size=None → RLKVPolicy 直接 int(None)
- 修复:window 默认条件改为 `is_rlkv_entry or policy == "rlkv"`(语义一致:
  只要 policy 是 rlkv,未显式给 window 就用官方默认)
- 教训:同一语义(RLKV 默认 window)应挂在 policy 上而非入口标志上,
  两个入口迟早会分叉

## Gate S0 验证

```text
单测:70 全绿(新增 14:确定性/quantile 语义/fallback/shape/与 duo 差异)
老入口(--enable-rlkv-inference, adapter-load-path):
  Tf=408878(2.0x vs T0),server fired up,生成正常
新入口(--enable-headkv --headkv-policy rlkv):
  Tf=408880(±2 profiling 噪声),server fired up,生成正常
双入口输出逐字一致("Paris")——确定性 mask 保证
GPU util:请求处理时 72%,idle 时 0%(正常,非异常)
```

## 面试素材

- "双入口收敛":老入口兼容(参数不变)+ 新入口显式 policy,runtime 无分支
- 确定性:原 loader 的 np.random.uniform 微扰 → 稳定 top-k/quantile,
  双入口输出可复现(实验可追问:同一 adapter 双入口输出逐字一致)
