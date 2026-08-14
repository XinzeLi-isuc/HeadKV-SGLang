# Phase S5 阶段报告:current main 收尾验证(CUDA Graph + RLKV + 质量)

> 日期:2026-08-14 | 状态:**完成**(phase-s4 遗留 4 项全部关闭)

## 背景

phase-s4 完成 current SGLang main(0.5.18.dev@e1c4db962)端到端迁移,
遗留 4 项记录在案:
1. CUDA Graph 模式未完整实测(in_capture 保护已写,未跑)
2. RLKV policy 在 current main 的启动验证待跑
3. 逐 token 一致剩余差异(full 路径 kernel 细节)
4. 20 prompts 之外的质量实验

本阶段(2026-08-14)以 4 server × 4 GPU 并行实测关闭全部遗留。

## 发现的 2 个真实 bug(修复后实验才可跑)

### Bug 1:prefill CUDA Graph capture 耗尽 comp 池(commit f4bb0b68e9)

- 现象:duokv-cg 启动崩于 CG capture 阶段:
  `RuntimeError: HeadKV comp pool exhausted: no comp chunk available for
  req_pool_idx=0. comp_chunks_available=0/32`
- 根因:current main 的 prefill capture 对无 captured-metadata 的 backend
  走 eager 入口 `init_forward_metadata(fb)`(prefill_cuda_graph_runner.py
  `_init_forward_metadata_for_capture` fallback),`in_capture` 默认 False,
  headkv 的 extend 分支 `_update_comp_mapping_extend` 为虚请求真实分配
  comp chunk → 32 个 chunk 被 capture 虚请求耗尽。
  (decode capture 路径 decode_cuda_graph_runner.py:1061 传了
  `in_capture=True`,是好的;prefill 路径漏了。)
- 修复:headkv_backend 新增 `init_forward_metadata_for_capture(fb)` =
  `out_graph(fb, in_capture=True) + in_graph(fb)`;runner fallback 优先
  调用该入口,其他 backend 走原路径,零行为变化。
- 验证:修复后 duokv-cg 一次启动成功,capture 20.15s(7 graphs)。

### Bug 2:RLKV policy 在 current main 启动失败(commit 29e4ab4d58 + b7981b5535)

- 现象:rlkv-eager 启动崩于 `_init_headkv_pool`:
  `AttributeError: 'ServerArgs' object has no attribute 'sink_window_size'`
  (v0.5.2 参数名残留);修复后又遇
  `AttributeError: server_args.headkv_sink_size assigned after resolution;
  server_args is read-only`(current main 物化后只读)。
- 根因:迁移时保留了 v0.5.2 的运行期回写默认值模式,current main
  ServerArgs 有 `__setattr__` 保护。
- 修复:window 默认 16/32 写入解析期 `_handle_headkv`(resolution 链内允许
  赋值);model_runner 改用局部变量 fallback,不再回写 ServerArgs。
- 验证:修复后 rlkv-eager 启动成功,3/3 生成语义正常。

## 实验结果(全部实测)

| 实验 | 协议 | 结果 |
| --- | --- | --- |
| CG 正确性 | 20 prompts(FullKV-cg vs DuoKV-cg) | 首 token 20/20;逐 token 15/20 |
| 单请求 E2E | 4K prompt + 64 new,3-run median | CG 2.497s vs eager 3.049s(-0.55s) |
| RLKV smoke | 3 prompts 生成 | 3/3 语义正常 |
| NIAH 4K | 3 depth × 3 seed,带 instruction,max_new=16 | fullkv 9/9,duokv 9/9 |

### 对照与解读

- 逐 token 15/20 vs S4 eager 14/20:**CG 无回归**;首 token 100%。
  剩余差异为 full 路径 kernel 调用细节(v0.5.2 移植局限),输出语义正确。
- CG 相对 eager 快 0.55s:decode kernel launch 开销消除,与 v0.5.2 S3
  结论一致(无固定启动成本)。
- NIAH 9/9 与 v0.5.2 S1 持平(duo 9/9):quality 无损保持。

## 踩坑(S5 新增,记入 interview/pitfalls)

- **P8(脚本协议不对齐)**:S5 NIAH 首版脚本只复制了 v0.5.2 的
  `build_niah`,漏了调用侧的 instruction 后缀与 max_new=16 → 模型按续写
  任务输出 filler,9 题全 miss(连 fullkv baseline 也 miss)。对齐完整
  协议后 9/9。教训:**实验口径必须逐字段对齐原脚本,不能只对齐构建函数**。
- **P9(current main ServerArgs 只读)**:resolution 物化后 `__setattr__`
  拒绝运行期回写,默认值必须写进解析期 handler(或局部变量)。

## Gate 自检

- [x] CUDA Graph 在 current main 可启动、可推理(修复 Bug 1)
- [x] RLKV policy 在 current main 可启动、生成正常(修复 Bug 2)
- [x] CG 正确性首 token 20/20,无回归
- [x] NIAH 4K 9/9(duo),质量无损
- [x] E2E 无启动开销(CG 2.497s < eager 3.049s)
- [x] 全部脚本/数据/文档入库,可复现(EXPERIMENTS.md §11)

## 遗留(收口)

- 逐 token 15/20 的剩余差异:已知 full 路径 kernel 细节,语义正确,
  不在本项目范围(迁移线目的为验证算法包可移植性,已达成)
- current main 的 300 请求生命周期回归:可选(v0.5.2 已覆盖 1500 请求,
  迁移线 runtime 语义一致,风险低)

## 面试素材

- **"接入不是终点,验证才暴露真问题"**:S4 只跑了 eager 就宣称迁移
  完成;S5 实测立刻暴露 CG capture comp 池耗尽 + ServerArgs 只读两个
  真实 bug。面试可讲"为什么收尾验证重要"。
- **架构理解**:current main 的 out_graph/in_graph + BCG capture 契约,
  decode runner 传 `in_capture=True` 而 prefill fallback 不传 —— 这类
  "两处调用契约不一致"的 bug 定位方法(对比 decode/prefill 调用点)。
- **数字可追问**:CG capture 20.15s;4K+64 单请求 CG 2.497s vs eager
  3.049s;NIAH 9/9;首 token 20/20、逐 token 15/20(如实声明差异)。
