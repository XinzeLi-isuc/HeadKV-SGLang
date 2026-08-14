# HeadKV-SGLang (DuoKV-SGLang)

将 DuoAttention 的 Retrieval/Streaming KV-head pattern 接入 RLKV 的 SGLang
head-reallocation runtime:Retrieval (Full) Heads 保存完整历史,Streaming
(Compact) Heads 仅保存 sink + recent 窗口,节省的 KV 显存重新分配给 Full Pool,
从而提升 max_total_tokens 与连续批处理并发容量。

> 求职(AI Infra 推理框架/推理加速)项目,2026-08。**S 级完成**(双 policy 同一
> runtime + CUDA Graph 深化)+ **E1-E5 实验标注体系**(容量/上下文/并发/
> 在线/质量,与官方 DuoAttention 同协议对照)。
> 设计:`DESIGN.md`;升级计划:`HeadKV-SGLang_S级升级计划书.md`;
> 实验复现:`EXPERIMENTS.md`(§11 迁移线、§12 E1-E5 标注体系)。

## 结果速览(实测,A6000, Meta-Llama-3.1-8B-Instruct)

```text
KV token 容量:  T0=204824 → Tf=782432(ratio 0.25, 3.82x)
               1.94x(ratio 0.5),与预算公式逐位吻合
质量:          NIAH 4K 9/9 无损(Full/Official Duo/HeadKV 三系统一致)
               LongBench narrativeqa F1 32.48 vs Full 31.49(官方协议 30 条)
生命周期:      1500 混合请求零失败,allocator 无泄漏
单请求开销:    eager 固定 0.33s(启动成本,不随长度) → CUDA Graph 消除 90%+
               (4K 绝对差 +0.334s → +0.030s)
短请求吞吐:    eager -22% → CUDA Graph +3.7%(4.76 vs 4.59 req/s)
decode 吞吐:   CUDA Graph 下 94~165 token/s(eager 约 20 量级)
双 policy:     同一 runtime 支持 DuoAttentionPolicy 与 RLKVPolicy(输出一致,
               同 effective ratio 质量对照,head 分布 Jaccard=0.446)
E1-E5 标注体系(2026-08-14, splits=32 修复后全量重跑):
  容量     : full/comp pool bytes 4 点标注(exp1_capacity.csv)
  上下文   : 32K prefill HeadKV 快 27%(6.68s→4.85s, exp2_context.csv)
  并发     : 16K×24 容量红利窗口 HeadKV 快 9%(exp3_concurrency.json)
  在线     : memory-bound 吞吐 +26%(0.53→0.67 req/s, exp4_online.json)
  质量     : 官方 DuoAttention 同协议三系统对照(exp5_quality.csv)
```

## 项目进度

- [x] Phase 0-8:MVP(A 级,2026-08-13,Gate 0-7)
- [x] Phase S0:双 policy 统一入口(Gate S0,2026-08-14;70 单测)
- [x] Phase S1/S2:双 policy 对照 + head 分布分析(Gate S1/S2)
- [x] Phase S3:CUDA Graph 开启(Gate S3;0.33s 消除 90%+)
- [x] Phase S4:current SGLang main 端到端迁移(Gate S4,2026-08-14;
      sglang 0.5.18.dev@e1c4db962 上 E2E 打通, T0=204698→Tf=397108;
      独立 env headkv-main;headkv 算法包零改动)
- [x] Phase S5:current main 收尾验证(2026-08-14;S4 遗留 4 项全关:
      CG 实测 + RLKV 启动 + 质量;修复 CG capture comp 池耗尽与
      ServerArgs 只读 2 个真实 bug,详见 EXPERIMENTS.md §11)
- [x] Phase S6:文档交付
- [x] E1-E5 实验标注体系(2026-08-14;容量/上下文/并发/在线/质量,
      与官方 DuoAttention 同协议对照;发现并修复 triton
      num_kv_splits=8 长上下文退化,详见 EXPERIMENTS.md §12)

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

- Prefix-Radix Cache / speculative decoding / TP>1 / PD 分离 / KV offload /
  FP8-INT4 KV / head 动态分类 / upstream PR

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
