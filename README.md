# HeadKV-SGLang (DuoKV-SGLang)

将 DuoAttention 的 Retrieval/Streaming KV-head pattern 接入 RLKV 的 SGLang
head-reallocation runtime:Retrieval (Full) Heads 保存完整历史,Streaming
(Compact) Heads 仅保存 sink + recent 窗口,节省的 KV 显存重新分配给 Full Pool,
从而提升 max_total_tokens 与连续批处理并发容量。

> 求职(AI Infra 推理框架/推理加速)项目,2026-08。MVP 已完成(Gate 0-7 全部通过)。
> 设计文档:`DESIGN.md`;计划书:`HeadKV-SGLang_修订版项目计划书.md`。

## 结果速览(实测,A6000, Meta-Llama-3.1-8B-Instruct)

```text
KV token 容量:  T0=204824 → Tf=385072(ratio 0.5, 1.88x)
               最高 3.82x(ratio 0.25),与预算公式逐位吻合
质量:          NIAH 4K 9/9 无损;LongBench narrativeqa/2wikimqa F1 持平或略高
吞吐:          单请求 -15~24%(eager 双路开销);长上下文高并发场景持平
KV 占用:       等量负载 token usage FullKV 0.31 vs DuoKV 0.16(减半)
```

## 项目进度

- [x] Phase 0 环境冻结与基线(Gate 0:FullKV + Official Duo smoke)
- [x] Phase 1 RLKV 调用链逆向(Gate 1:10 问 + 生命周期图)
- [x] Phase 2 HeadPolicy 抽象(Gate 2:33 单测,确定性二值化)
- [x] Phase 3 双池接入(Gate 3:1.94x 容量,启动日志)
- [x] Phase 4 attention 语义正确性(Gate 4:45 单测 + 20/20 E2E 一致)
- [x] Phase 5 物理双池与容量 Gate(Gate 5:3.82x 单调,comp 耗尽 fail-fast)
- [x] Phase 6 生命周期与 continuous batching(Gate 6:1500 请求零失败)
- [x] Phase 7 正式实验(Exp A-E)
- [x] Phase 8 整理交付

## 代码结构

```text
fork: ~/rlkv/sglang @ 973b5e41 + 分支 feat/headkv-duo
├── python/sglang/srt/headkv/          # 新增:纯算法,零 SGLang 内部依赖
│   ├── config.py      HeadKVConfig(校验/window 优先级)
│   ├── policy.py      HeadPolicy 抽象 + 工厂
│   ├── duo_policy.py  DuoAttention 加载 + 确定性 top-k/threshold 二值化 + GQA 校验
│   ├── rlkv_policy.py RLKV adapter 兼容(去随机微扰)
│   ├── manual_policy.py
│   ├── partition.py   TP-local 划分(完备且不相交)
│   └── budget.py      双池容量预算(计划书公式 + 边界)
├── python/sglang/srt/server_args.py   # 修改:--enable-headkv 等 7 参数
├── python/sglang/srt/model_executor/model_runner.py  # 修改:_init_headkv + 双池接入
├── python/sglang/srt/layers/attention/head_realloc_backend.py  # 修改:window 优先 + fail-fast
└── test/srt/headkv/                   # 新增:56 个单测(CPU)
```

项目目录(本仓库):
```text
HeadKV-SGLang/
├── DESIGN.md / 计划书
├── docs/headkv/        env / architecture / rlkv_callgraph / known_limitations
├── docs/reports/       phase0-7 阶段报告
├── docs/interview/     面试素材(pitfalls / narrative)
├── benchmarks/headkv/  实验脚本
├── artifacts/          smoke/实验原始数据
├── results/            capacity / throughput / online_serving / quality CSV
└── figures/            容量图表
```

## 组件归属(README 必答)

**复用了 RLKV 的 runtime 组件**(Kurt232/rlkv-sglang-v0.5.2):
- `HeadReallocKVPool`(每层 full/comp 双 buffer + _fused_kv_write)
- `HeadReallocAllocator`(full 池 + per-request comp chunk 管理)
- `HeadReallocAttnBackend`(full/comp 双路 attention、sink/recent 环形映射、
  CUDA Graph window indices 机制)
- DuoAttention 官方 pattern 与官方 oracle(mit-han-lab/duo-attention)

**新增(本项目)**:
- `HeadPolicy` 抽象:mask 加载/二值化与 runtime 解耦(runtime 只消费
  mask + sink/recent)
- `DuoAttentionPolicy`:官方 pattern 确定性二值化(稳定 top-k 替代随机微扰)、
  GQA 维度校验(KV 粒度禁止二次 OR)、window 优先级解析(禁回落 16/32 默认)
- `partition.py` / `budget.py`:TP 划分与容量预算(含 F=0/C=0/不可能配置边界)
- CLI:`--enable-headkv` 等 7 参数,与 `--enable-rlkv-inference` 互斥

**修改(修复)**:
- comp 池耗尽:静默写 dummy → fail-fast RuntimeError(含池状态)
- backend window 读取:优先 policy 解析值(HeadKV 下 Duo 128/256 不被
  RLKV 默认 16/32 覆盖)
- 预算公式:统一计划书公式(现状带 T0 下限 → 显式无静默 clamp)

## 已验证

- 模型:Meta-Llama-3.1-8B-Instruct(bf16, GQA);TP=1;eager;page_size=1
- 场景:4K-16K 上下文、单请求/并发/连续批处理、NIAH/LongBench

## 未支持(MVP 范围外)

- current SGLang main 迁移 / Prefix-Radix Cache / CUDA Graph 开启 /
  speculative decoding / TP>1 / PD 分离 / KV offload / FP8-INT4 KV /
  head 动态分类 / upstream PR

## 运行

```bash
export PATH=/home/lixinze/miniconda3/envs/rlkv-eval/bin:$PATH
export NO_PROXY="127.0.0.1,localhost"   # 本机代理会劫持本地回环请求
# FullKV baseline(同 triton backend,公平对比)
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
  --model-path ~/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct \
  --port 30000 --mem-fraction-static 0.85 --max-running-requests 32 \
  --disable-radix-cache --disable-cuda-graph --attention-backend triton
# DuoKV
... --enable-headkv --headkv-policy duo \
    --headkv-pattern-path <pattern_dir> --headkv-full-head-ratio 0.5
```

测试:`python -m pytest test/srt/headkv/ -q`(56 用例,纯 CPU)
