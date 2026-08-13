# 已知限制与待修问题(known_limitations.md)

> Phase 1 交付物,2026-08-13。基于代码逆向发现的现状问题,
> 与计划书修订表/设计文档 §1.4/§4.5 对应。全部有真实行号。

## K1. comp 池耗尽静默降级(计划书 Phase 5 要点 5 / 设计 §4.5)

- 位置:allocator.py `alloc_comp_window`(L349-354 返回 0);
  backend `_get_comp_base`(L202/L215 返回 0);
  `_update_comp_mapping_extend` L287-288 `continue`(静默跳过);
  `_update_comp_mapping_decode` L259-263(写 0)
- 后果:超过 max_comp_chunks 的请求 comp heads 读 dummy slot,输出错误无提示
- 修复:fail-fast(RuntimeError + comp 池状态日志),见 DESIGN §4.5

## K2. mask 二值化不可复现(计划书修订 D1)

- 位置:model_runner.py `_load_rlkv_head_masks` L1764
  `adapter_weight += np.random.uniform(0, 1e-6, ...)`
- 后果:同 sparsity 两次运行 mask 不同 → 实验不可复现
- 修复:稳定排序 (score, layer_id, head_id) 确定性 top-k,见 DESIGN §3.3

## K3. extend 窗口外 token 写 comp dummy slot

- 位置:backend `_update_comp_mapping_extend` L315-318(窗口外 mapping=0);
  memory_pool.py `set_kv_buffer`(comp_loc=0 → 写 dummy)
- 后果:无效写入开销 + dummy 槽被反复覆盖(注释自认 harmless, L275)
- 评估:MVP 保留(开销小),容量实验的 pool byte accounting 需排除 dummy

## K4. extend mapping 更新为 Python 循环

- 位置:backend `_update_comp_mapping_extend` L280-320 `for i in range(bs)`
- 后果:bs 大时(连续批处理)host 开销明显;decode 路径已向量化(L217-264),
  extend 路径未向量化,不对称
- 评估:MVP 保留;性能实验中若 extend 占比高,需向量化(候选优化点,面试素材)

## K5. chunked extend 的 comp 落盘语义未验证

- 位置:backend L266-320(recent 判定 `max(sink, seq_len - recent)`)
- 风险:同一 request 多次 extend 时,recent 窗口滚动,旧 recent 槽被覆盖;
  与一次性 extend 的最终落盘是否一致需单测验证(设计 §4.2 已列)

## K6. 预算公式与 fork 现状不一致(计划书修订 D4)

- 位置:model_runner.py L1571-1575 `size_full = max(..., max_total_num_tokens)`
- 差异:现状带 T0 下限,设计统一为计划书公式(无静默 clamp),见 DESIGN §3.6

## K7. RLKV 默认 window(16/32)与 Duo 官方(128/256)不一致(计划书修订 D3)

- 位置:server_args.py L331-332
- 修复:HeadKV 模式默认读 pattern config.json,命令行显式覆盖,见 DESIGN §3.1

## K8. max_running_requests 默认 48 静默生效(计划书修订 D5)

- 位置:server_args.py L748-749
- 修复:HeadKV 模式强制显式指定,见 DESIGN §3.1

## K9. TP 支持现状

- `_load_rlkv_head_masks` 有 TP slice(L1783-1797),`HeadReallocKVPool/Allocator`
  接口含 TP 语义,但 head_realloc backend 的 Q 头展开按 TP 局部计算
- MVP 仅 TP=1;TP>1 未验证,列入"未支持"清单(README/交付说明)

## 未支持清单(设计 §7.1 固定约束之外)

- current SGLang main 迁移 / Prefix-Radix Cache(RLKV 强制关闭,L562)/
  speculative decoding / TP-PP-DP / PD 分离 / KV offload / 动态 head 分类 /
  FP8-INT4 KV / 自定义融合 kernel / upstream PR
