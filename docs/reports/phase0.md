# Phase 0 阶段报告:环境冻结与 FullKV 基线

> 日期:2026-08-13 | 状态:**完成(Gate 0 全绿)**

## 目标(计划书 §6 Phase 0)

1. 固定 RLKV fork commit ✓
2. 安装匹配依赖 ✓(rlkv-eval 零改动直接可用)
3. 主模型 + DuoAttention pattern 本地就位 ✓
4. 跑通 Vanilla SGLang FullKV ✓(2026-08-13 21:32)
5. 跑通官方 DuoAttention 单请求生成(进行中)
6. 固定 20 条 correctness prompts ✓

## 环境冻结结论(核心交付)

- **rlkv-eval env 零改动直接可用**:py3.10.20 / torch 2.8.0+cu128 / triton 3.4.0 /
  flashinfer 0.3.1 / numpy 2.2.6 / transformers 4.56.1 / modelscope 1.39.1
- sglang 0.5.2 已 editable 指向 fork(commit 973b5e41)
- 模型:modelscope 本地缓存,30G 完整,32L/32Q/8KV(GQA),bf16
- pattern:官方 32×8 已 clip [0,1],config sink=128/recent=256/threshold=0.5

## 关键成果:FullKV smoke 通过 + Official Duo smoke 通过

```text
[FullKV] server 就绪:The server is fired up and ready to roll!
         4K prompt → 32 tokens:elapsed 1.44s,输出语义正确
         T0(baseline token capacity):204824(干净 GPU 上;有残留进程时 169437)
[Duo]    pattern: sink=128 recent=256 threshold=0.5,shape [32,8] 值域 [0,1]
         官方 DuoAttention eval patch 启用成功,生成 3.42s
         "What is the capital of France?" → "Paris"
```

**T0 = 204824 是 Phase 3 预算公式的关键输入**,已在 env.md 记录测量条件。

## 踩坑(详见 docs/interview/pitfalls.md)

| # | 问题 | 根因 | 修复 |
| --- | --- | --- | --- |
| P1 | warmup 502 三小时 | http_proxy=127.0.0.1:7884 劫持本地请求,无 NO_PROXY | export NO_PROXY |
| P2 | ninja not found | python 绝对路径调用,PATH 无 env bin | export PATH |
| P3 | duo_attn import 失败 | transformers 4.56 移除旧导出(List/CrossEntropyLoss) | 修 4 个文件 import |
| P4 | device_map 需 accelerate | rlkv-eval 未装 accelerate | 改 .to("cuda:1") |

排查方法论亮点(面试素材):faulthandler SIGABRT 抓线程栈证明 scheduler 健康
→ 转向环境层审查 → curl --noproxy 对照实锤代理劫持。全程未重装任何包。

## Gate 0 状态(全绿,2026-08-13 21:46)

- [x] FullKV 稳定生成(4K smoke 1.44s)
- [x] Official DuoAttention 同模型/pattern 生成(3.42s,"Paris")
- [x] pattern shape 与 num_layers × num_kv_heads 一致(32×8 已核实)

## 下一步

1. 干净 GPU 上复测 T0=204824(正式基线存档,容量实验前置)
2. Phase 1:RLKV 调用链逆向(10 问,产出 rlkv_callgraph.md / architecture.md)
