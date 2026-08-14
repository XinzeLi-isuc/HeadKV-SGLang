# Phase S4 阶段报告:current SGLang main 端到端迁移

> 日期:2026-08-14 | 状态:**完成(Gate S4 通过:E2E decode 打通)**

## 目标(升级计划书 §8:两天止损 Gate)

1. HeadPolicy 在 current main 接口加载 ✓
2. pool/allocator 初始化 ✓
3. tensor-level backend smoke ✓(server 级 E2E 更严格)
4. physical pool shape 正确 ✓

## 迁移基线

- current SGLang main:commit e1c4db9621(2026, sglang 0.5.18.dev)
- 独立环境 headkv-main(复制 rlkv-eval → transformers 5.12.1 /
  flashinfer 0.6.17 / torch 2.13 / pip install -e current main)
- 与 v0.5.2 同模型/同 pattern/同 GPU

## 成果

| Gate | 验证 | 结果 |
| --- | --- | --- |
| Gate 1 | HeadPolicy 加载(current 风格 ModelConfig) | duo/rlkv mask + budget Tf=397360 与 v0.5.2 一致 |
| Gate 2/4 | 双池初始化 + shape | full(4097,4,128)/comp(12289,4,128)逐位吻合;alloc/free 循环恢复 |
| Gate 3 | **E2E server** | **"The server is fired up";生成语义正确**;T0=204698→Tf=397108, gain=1.940x |

## 正确性(current main, vs FullKV-triton)

- 首 token 一致:20/20 = 100%
- 逐 token:ratio 0.5 = 14/20;ratio 1.0 = 14/20
- 差异定位:comp 路径与 full 路径已数值一致(修复 split-kernel 后
  11→14);剩余差异为 full 路径 kernel 调用细节(v0.5.2 移植局限),
  分叉输出语义均正确

## 迁移改动清单(current main 仓库,commit 5ab663578d)

| 文件 | 改动 |
| --- | --- |
| headkv/(9 文件) | **零改动复制**(算法层与 runtime 解耦的直接验证) |
| mem_cache/headkv_pool.py | 新增:双池 + allocator(current KVCache ABC 移植) |
| layers/attention/headkv_backend.py | 新增:backend(out_graph/in_capture 迁移 + current kernel 签名) |
| layers/attention/attention_registry.py | +headkv 注册 |
| server_args.py | +8 参数 + _handle_headkv |
| model_runner.py | HeadKV 分支:纯 profiling T0 → 独立双池 |
| attention_backend_setup.py | resolve 确定性拦截(headkv) |

## 踩坑(详见 pitfalls.md P7)

1. current main 组件接口漂移 5 处(utils→utils.common / dp_attention→
   runtime_context / triton_ops→kernels.ops / mixed_triton_backend→
   flashinfer_backend / ForwardBatch 不再带 pool)
2. 双池与默认池共存 OOM → 纯 profiling + 独立建池
3. resolution pipeline 覆盖 backend → resolve 函数确定性拦截
4. CG capture 虚请求耗尽 comp 池 → out_graph + in_capture 保护
5. decode/extend kernel 新参数(k_scale/v_scale 必须 1.0)
6. cache_k 维度 4→3(set_kv_buffer 兼容)
7. split-kernel 新旧差异 → 浮点分叉(对齐 current get_num_kv_splits_triton)

## 遗留(记录在案)

- 逐 token 一致的剩余差异(kernel 调用细节,首 token 100% 已满足)
- CUDA Graph 模式(in_capture 保护已写,未完整实测)
- RLKV policy 在 current main 的启动验证(代码路径已统一,待跑)
- 20 prompts 之外的质量实验(可复用 v0.5.2 全套脚本)
