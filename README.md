# HeadKV-SGLang (DuoKV-SGLang)

将 DuoAttention 的 Retrieval/Streaming KV-head pattern 接入 RLKV 的 SGLang
head-reallocation runtime,使 Retrieval Heads 保存完整历史、Streaming Heads 仅保存
sink + recent,并把节省的显存重新分配给 Full Pool,从而提升 max_total_tokens 与
连续批处理并发容量。

> 本项目为求职(AI Infra 推理框架/推理加速)项目,设计文档见 `DESIGN.md`,
> 原始计划书见 `HeadKV-SGLang_修订版项目计划书.md`。

## 当前状态

- [x] 计划书(修订版)
- [x] 技术设计文档(DESIGN.md)
- [x] 环境核查:rlkv-eval env 零改动可用(见 docs/headkv/env.md)
- [x] Phase 0:环境冻结 + FullKV/Official Duo smoke(Gate 0 全绿,2026-08-13)
- [x] Phase 1:RLKV 调用链逆向(Gate 1 通过,2026-08-13;10 问全答 + 三者关系图)
- [x] Phase 2:HeadPolicy / Duo loader / partition / budget(Gate 2 通过,2026-08-13;33 单测)
- [ ] Phase 3:双池接入
- [ ] Phase 4:attention 语义正确性
- [ ] Phase 5:物理双池与容量 Gate
- [ ] Phase 6:request 生命周期与 continuous batching
- [ ] Phase 7:正式实验
- [ ] Phase 8:整理交付

## 目录结构

```text
HeadKV-SGLang/
├── DESIGN.md                        # 技术设计文档(接口/数据流/预算/测试/实验)
├── HeadKV-SGLang_修订版项目计划书.md # 原始计划书
├── docs/
│   ├── headkv/env.md                # 环境冻结记录
│   ├── interview/                   # 面试素材(踩坑/改动/叙事)
│   └── reports/                     # 阶段报告
├── benchmarks/headkv/               # 实验脚本与 prompts
└── artifacts/                       # smoke 日志与中间产物
```

代码实现在 `~/rlkv/sglang` fork 的独立分支 `feat/headkv-duo` 上
(RLKV SGLang v0.5.2,commit 973b5e41)。

## 关键基线

- 模型:Meta-Llama-3.1-8B-Instruct(GQA 32Q/8KV/32L,bf16,modelscope 本地缓存)
- Pattern:DuoAttention 官方 `lr=0.02-reg=0.05-ctx=1000_128000-multi_passkey10`
  (32×8,sink=128,recent=256,threshold=0.5)
- FullKV baseline(同 commit):`max_total_num_tokens = 204824`(T0)
- 环境:rlkv-eval(py3.10 / torch 2.8.0+cu128 / triton 3.4 / flashinfer 0.3.1)
