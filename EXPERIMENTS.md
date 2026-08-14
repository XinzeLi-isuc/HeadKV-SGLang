# HeadKV-SGLang 实验复现报告

> 最后更新:2026-08-14。本文档保证:按步骤执行即可复现全部实验数字。
> 所有脚本在 `benchmarks/headkv/`,原始数据在 `artifacts/`,汇总在 `results/`。

## 0. 环境与前置(必读)

```bash
# 仓库与分支
fork:  ~/rlkv/sglang @ 973b5e41 + 分支 feat/headkv-duo(含 S0-S3 全部改动)
项目:  ~/HeadKV-SGLang

# 环境(零改动,直接用 rlkv-eval)
export PATH=/home/lixinze/miniconda3/envs/rlkv-eval/bin:$PATH
export NO_PROXY="127.0.0.1,localhost"   # 必须:本机代理会劫持本地回环请求(P1)

# 模型与 pattern
模型:  ~/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct
Duo:   ~/duo-attention-ref/attn_patterns/Meta-Llama-3.1-8B-Instruct/lr=0.02-reg=0.05-ctx=1000_128000-multi_passkey10/
RLKV:  ~/rlkv/head_dist/rlkv/Llama-3.1-8B-R1/llama_lr1e-2_ep2_bs32_reg1e-3_tau0.5/

# 硬件: 2× A6000 48GB(GPU 0/1),driver 支持 cu128
```

启动模板(eager 或 CUDA Graph):

```bash
# FullKV baseline(必须 triton backend,与 DuoKV 同 kernel —— P3 kernel 混淆教训)
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
  --model-path <MODEL> --port 30000 --mem-fraction-static 0.85 \
  --max-running-requests 32 --disable-radix-cache --attention-backend triton \
  [--disable-cuda-graph]   # eager 加此项;CUDA Graph 实验去掉

# DuoKV
... --enable-headkv --headkv-policy duo --headkv-pattern-path <DUO> \
    --headkv-full-head-ratio 0.5 [--disable-cuda-graph]

# RLKV(双入口等价,S0 验证过输出逐字一致)
# 老入口:--enable-rlkv-inference --adapter-load-path <RLKV> [--rlkv-sparsity 0.5]
# 新入口:--enable-headkv --headkv-policy rlkv --headkv-pattern-path <RLKV> --rlkv-sparsity 0.5
```

## 1. 单元测试(70 用例,纯 CPU,~6s)

```bash
cd ~/rlkv/sglang && python -m pytest test/srt/headkv/ -q
# 70 passed
```

覆盖:config 校验/确定性二值化/window 优先级/budget 边界/partition 完备性/
pool shapes/byte accounting/allocator 生命周期/comp 耗尽 fail-fast/
extend-decode mapping 语义/ref_attention 窗口数学/RLKVPolicy 全项。

## 2. 容量实验(Exp A)

```bash
# 4 点 ratio sweep(GPU 0,记录启动日志的 Tf 与 max_total_num_tokens)
bash benchmarks/headkv/run_capacity_sweep.sh   # 0.25/0.75 两点
# 0.5 与 1.0 另测(见 artifacts/capacity_sweep.csv + capacity_sweep_gate3.log)
```

| ratio | full/compact | Tf 实测 | 增益 | 公式手算 |
| --- | --- | ---: | ---: | ---: |
| 0.25 | 64/192 | 782432 | 3.82x | 782432 |
| 0.50 | 128/128 | 397360 | 1.94x | 397360 |
| 0.75 | 192/64 | 269002 | 1.31x | 269002 |
| 1.00 | 256/0 | 204824 | 1.00x | 204824 |

公式:Tf = floor((T0×(F+C) − R×V×C) / F),T0=204824,V=sink+recent。

## 3. 正确性(Exp E 前置 + E2E)

```bash
# 20 条 prompts(短 prompt,FullKV-triton vs DuoKV,应逐 token 一致)
python benchmarks/headkv/run_correctness.py --fullkv-port 30000 --duokv-port 30001 \
  --prompts benchmarks/headkv/prompts_correctness.jsonl --max-new-tokens 64 \
  --out artifacts/correctness_e2e.json
# 预期:20/20 exact;首 token 20/20(实测 1.00)

# 4K 长 prompt smoke
python benchmarks/headkv/send_generate.py --port <PORT> --prompt-file artifacts/prompt_4k.txt \
  --max-new-tokens 32 --out artifacts/xxx.json
```

## 4. 生命周期压测(Phase 6)

```bash
python benchmarks/headkv/run_lifecycle_stress.py --port <PORT> --rounds 1000 \
  --workers 8 --out artifacts/lifecycle_stress_1000.json
# 预期:ok=1000/1000 failures=0;第二轮 500 吞吐无衰减(allocator 无泄漏)
```

## 5. 单请求与并发(Exp B/C)

```bash
python benchmarks/headkv/run_experiments.py --fullkv-port 30000 --duokv-port 30001 \
  --out artifacts/exp_bc.json                       # Exp B(4K/8K/16K)+ C(bs sweep)
python benchmarks/headkv/run_expc.py                # C 聚焦版(8K×16/32/48)
python benchmarks/headkv/run_expc_decodeheavy.py    # 4K×64×128
python benchmarks/headkv/run_expd.py                # online(memory-light/bound)
```

## 6. 质量(Exp E)

```bash
python benchmarks/headkv/run_niah_mini.py           # NIAH 4K,9 例
python benchmarks/headkv/run_longbench.py --fullkv-port 30000 --duokv-port 30001 \
  --max-samples 40 --out artifacts/longbench.json   # narrativeqa + 2wikimqa
```

## 7. S 级实验

```bash
# S0 双入口一致性:老入口(--enable-rlkv-inference)vs 新入口(--headkv-policy rlkv)
#   同 adapter/sparsity → 输出逐字一致,Tf 相同(±2 token)

# S1 双 policy 对照(同 effective ratio 0.5)
python benchmarks/headkv/run_s1_quality.py --duo-port 30040 --rlkv-port 30041 \
  --out artifacts/s1_quality.json

# S2 head 分布(Jaccard/heatmap)
python benchmarks/headkv/analyze_head_distribution.py

# S3 CUDA Graph:去掉 --disable-cuda-graph 启动,重复 §3/§5 的
#   20 prompts 正确性 + 4K/8K/16K E2E + 短请求吞吐
```

## 8. 主张补证实验(2026-08-14)

```bash
# 实验 A:长上下文有界差异(8K/16K 分叉率,需 FullKV-CG:30060 + DuoKV-CG:30061)
python benchmarks/headkv/run_expA.py
# 预期:首 token 10/10 一致;逐 token 8K≈0.15/16K≈0.07;分叉后语义正确

# 实验 B:NIAH 损失边界(需 4 server:fullkv 30060 / duo 0.25 30062 / 0.5 30061 / 0.75 30063)
python benchmarks/headkv/run_expB.py
# 预期:fullkv 6/6、duo 0.5/0.75 6/6、duo 0.25 2/6(损失边界)

# 实验 C+D:32K/64K 服务能力 + CG 并发(需 30060/30061)
python benchmarks/headkv/run_expCD.py
# 预期:64K duokv -32%(FLOPs 收益);CG bs=32 duokv -2%
```

## 9. S4 current-main 迁移(2026-08-14)

```bash
# 环境(独立,不碰 rlkv-eval)
cp -a ~/miniconda3/envs/rlkv-eval ~/miniconda3/envs/headkv-main
export PATH=/home/lixinze/miniconda3/envs/headkv-main/bin:$PATH
python -m pip install transformers==5.12.1 xgrammar==0.2.1 \
  "flashinfer_python[cu13]==0.6.17"
python -m pip install -e ~/sglang-main/python   # current main @ e1c4db962
# 注意:复制 env 后 pip shebang 仍指向旧 env,必须用 python -m pip

# 启动(current main 上 HeadKV, eager)
cd ~/sglang-main && export NO_PROXY="127.0.0.1,localhost" \
  && CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
  --model-path <MODEL> --port 30080 --mem-fraction-static 0.85 \
  --max-running-requests 32 --disable-cuda-graph \
  --enable-headkv --headkv-policy duo \
  --headkv-pattern-path <DUO> --headkv-full-head-ratio 0.5
# 预期: [HeadKV] ... T0=204698 Tf=397108 gain=1.940x; fired up; 生成正确

# 对照(FullKV-triton, current main)
CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server --model-path <MODEL> \
  --port 30081 --mem-fraction-static 0.85 --max-running-requests 32 \
  --disable-cuda-graph --attention-backend triton

# 正确性(20 prompts)
python benchmarks/headkv/s4_correctness.py
# 预期: 首 token 20/20; 逐 token 14/20(ratio 0.5/1.0 同, 差异为
#   kernel 调用细节, 分叉输出语义正确)
```

## 10. 数据文件索引

| 文件 | 内容 |
| --- | --- |
| artifacts/capacity_sweep.csv | 容量 4 点(0.25/0.5/0.75/1.0) |
| artifacts/capacity_sweep_gate3.log | 每点 [HeadKV] 日志 |
| artifacts/correctness_e2e.json | 20 prompts 对比 |
| artifacts/niah_mini.json | NIAH 9 例 |
| artifacts/longbench.json | LongBench 40×2 |
| artifacts/exp_bc.json / exp_c_concurrency.json / exp_c_decodeheavy.json | B/C |
| artifacts/exp_d_online.json | D |
| artifacts/lifecycle_stress_1000.json / round2 / s3_lifecycle_cg.json | 生命周期 |
| artifacts/s1_quality.json | S1 双 policy 质量 |
| artifacts/s2_head_stats.json | S2 Jaccard/分布 |
| results/*.csv | 汇总(capacity/throughput/online/quality) |
| figures/*.png | 容量/开销/并发/分布图 |

## 9. 复现验证清单

- [ ] 70 单测全绿
- [ ] 容量表 4 点与公式逐位吻合
- [ ] 20/20 E2E 一致(eager + CUDA Graph 两种模式)
- [ ] 1000+500 请求零失败
- [ ] NIAH 9/9、LongBench F1 与报告一致
- [ ] S0 双入口输出一致
- [ ] S1 容量只由 F/C/V 决定;Jaccard≈0.446
- [ ] S3:4K 绝对差 ≤0.05s;短请求吞吐 ≥4.5 req/s

## 11. S5 收尾:current main 剩余验证(2026-08-14)

> 仓库:`~/sglang-main`(sglang 0.5.18.dev@e1c4db962 + headkv 迁移),env `headkv-main`。
> 覆盖 phase-s4 报告遗留的 4 项:CUDA Graph 实测 / RLKV 启动验证 /
> 逐 token 差异收尾 / 质量实验。

### 11.1 修复(2 个独立问题, 3 个 commit)

| commit | 问题 | 修复 |
| --- | --- | --- |
| f4bb0b68e9 | prefill CG capture 虚请求耗尽 comp 池(comp_chunks_available=0/32 崩溃) | headkv_backend 新增 `init_forward_metadata_for_capture`(out_graph 以 in_capture=True 跳过 comp 分配);prefill_cuda_graph_runner fallback 优先调用,其他 backend 不变 |
| 29e4ab4d58 + b7981b5535 | RLKV policy 启动 AttributeError:ServerArgs 无 sink_window_size / 物化后只读 | window 默认 16/32 写入解析期 `_handle_headkv`;model_runner 局部变量 fallback |

### 11.2 实验(4 server × 4 GPU, A6000, 同一模型/pattern)

```bash
export PATH=/home/lixinze/miniconda3/envs/headkv-main/bin:$PATH
export NO_PROXY="127.0.0.1,localhost"
# 4 个 server(见 benchmarks/headkv/current_main/start_server.sh)
#   fullkv-cg:30090 (GPU0)   duokv-cg:30091 (GPU1, duo ratio 0.5)
#   duokv-eager:30092 (GPU2) rlkv-eager:30093 (GPU3, sparsity 0.5)
python benchmarks/headkv/current_main/s5_cg_correctness.py   # 20 prompts, 30090 vs 30091
python benchmarks/headkv/current_main/s5_e2e_overhead.py     # 3-run median, 30091 vs 30092
python benchmarks/headkv/current_main/s5_rlkv_smoke.py       # 3 prompts, 30093
python benchmarks/headkv/current_main/s5_niah.py             # 9 题 4K, 30090 vs 30091
```

### 11.3 结果

| 实验 | 结果 | 对照 |
| --- | --- | --- |
| CUDA Graph 正确性 | 首 token 20/20;逐 token 15/20 | eager 14/20(S4)→ CG 无回归,差异为已知 full 路径 kernel 细节 |
| 单请求 E2E(4K+64) | CG 2.497s vs eager 3.049s(**-0.55s**) | v0.5.2 S3:CG 绝对差 +0.030s,current main 无启动开销 |
| RLKV policy 启动+生成 | 3/3 语义正常(sparsity 0.5, window 16/32) | 与 v0.5.2 RLKV 行为一致 |
| NIAH 4K(带 instruction) | fullkv 9/9, duokv 9/9 | v0.5.2 S1:duo 9/9,rlkv 9/9,无损保持 |

> 踩坑:S5 NIAH 首版脚本漏了 v0.5.2 的 instruction 后缀
> ("\n\nWhat is the special magic number...")与 max_new=16,
> 模型按续写任务输出 filler 导致 miss;对齐 S1 协议后 9/9。
> 实验口径必须逐字段对齐原脚本,不能只对齐"构建函数"。

### 11.4 数据文件(新增)

| 文件 | 内容 |
| --- | --- |
| artifacts/s5_cg_correctness_main.json | current main CG: 20 prompts 首 token/逐 token |
| artifacts/s5_e2e_overhead_main.json | CG vs eager 3-run E2E |
| artifacts/s5_rlkv_smoke_main.json | RLKV 3 prompts 生成 |
| artifacts/s5_niah_main.json | NIAH 4K 9 题 fullkv/duokv |
| benchmarks/headkv/current_main/ | S5 全部脚本 + start_server.sh |

## 12. 实验标注体系 E1-E5(2026-08-14, 全部 splits=32 下有效)

> 统一协议: GPU=A6000, model=Meta-Llama-3.1-8B-Instruct(bf16, GQA),
> mem-fraction=0.85, TP=1, v0.5.2(feat/headkv-duo)主数据线。
> **重要**: 发现并修复 triton attention 长上下文退化 —— 默认
> `--triton-attention-num-kv-splits 8` 在 ≥4K 复杂文本下生成退化
> (重复 token 循环), splits=32 后 4K-31K 全域恢复(与 flashinfer/官方
> HF 输出一致)。E2/E3/E4/E5-LongBench 全部在 splits=32 下重跑。

### E1 KV Capacity(results/exp1_capacity.csv)

| effective ratio | full pool GB | comp pool GB | max_total_tokens | max_running |
| --- | --- | --- | --- | --- |
| 0.25 | 25.639 | 1.208 | 782432 | 32 |
| 0.50 | 26.041 | 0.805 | 397360 | 32 |
| 0.75 | 26.444 | 0.403 | 269002 | 32 |
| 1.00 | 26.847 | 0.000 | 204824 | 32 |

### E2 Context Length(results/exp2_context.csv, 3-run median, CG)

| ctx | fullkv TTFT | headkv TTFT | Δ | fullkv TPOT | headkv TPOT |
| --- | --- | --- | --- | --- | --- |
| 4K | 0.448s | 0.448s | 0% | 0.0229s | 0.0234s |
| 8K | 0.963s | 0.968s | +1% | 0.0234s | 0.0237s |
| 16K | 2.357s | 2.087s | -11% | 0.0244s | 0.0241s |
| 32K | 6.680s | 4.854s | **-27%** | 0.0265s | 0.0252s |

> 长上下文 prefill 收益: comp heads 只处理 sink+recent 窗口, 32K 时
> HeadKV prefill 快 27%(稀疏计算的实际收益)。

### E3 Batch/Concurrency(artifacts/exp3_concurrency.json)

| ctx | 关键点 | fullkv med | headkv med | Δ |
| --- | --- | --- | --- | --- |
| 8K | bs=96 | 41.87s | 40.45s | -3% |
| 16K | bs=24 (393K tokens, 超 FullKV T0=204K 但 < HeadKV Tf=397K) | 37.53s | 34.19s | **-9%** |
| 16K | bs=64 | 78.0s (P50>60s 不可用) | 92.3s | 均排队 |

> 容量红利窗口: total KV ∈ (FullKV T0, HeadKV Tf] 时 HeadKV 不排队而
> FullKV 排队(16K×24 快 9%)。超 HeadKV 容量后两者均排队, HeadKV
> 无优势(双池 decode 开销略高)。SGLang 调度器排队机制使"失败"不出现,
> 以延迟爆炸为不可用判据。

### E4 Online Serving(artifacts/exp4_online.json, 60s 流, 8 workers)

| workload | 指标 | fullkv | headkv |
| --- | --- | --- | --- |
| memory_light (512 tok) | req/s | 7.35 | 7.30 |
| | TTFT p50/p95 | 0.322/0.362s | 0.258/0.331s |
| | TPOT p50/p95 | 0.0234/0.031s | 0.0267/0.0312s |
| memory_bound (16K tok, 128 gen) | req/s | 0.53 | **0.67 (+26%)** |
| | output tok/s | 68.3 | **85.3 (+25%)** |
| | TTFT p50/p95 | 7.008/11.028s | 7.028/11.031s |
| | TPOT p50/p95 | 0.0774/0.111s | **0.0713/0.106s** |

### E5 Quality(results/exp5_quality.csv, 官方协议 30 条)

NIAH 4K(9 题 magic 协议): Full(HF) 9/9, Official Duo(HF) 9/9,
FullKV(SG) 9/9, HeadKV(SG) 9/9 —— **全部无损**。

LongBench F1:

| task | Full (HF) | FullKV (SG) | Official Duo (HF) | HeadKV (SG) |
| --- | --- | --- | --- | --- |
| narrativeqa | 31.49 | 31.49 | 28.26 | **32.48** |
| 2wikimqa | 21.35 | 20.38 | 20.98 | 18.89 |

> narrativeqa 无损(HeadKV 32.48 ≥ Full 31.49); 2wikimqa 相对官方
> duo 掉 2.1 F1(multi-hop 检索对 streaming head 更敏感, 30 条样本)。
> 官方 HF 与 SGLang 端 fullkv 完全一致(narrativeqa 31.49=31.49),
> 验证协议对齐有效。

### 关键发现(面试素材)

1. **triton attention num_kv_splits=8 长上下文退化**(默认参数 bug):
   复杂文本 ≥4K 即退化, splits=32 修复, 4K-31K 全域与 flashinfer/
   官方一致。诊断链: 官方 HF vs SGLang 输出对照 → flashinfer 对照 →
   split 数假设 → 参数验证。
2. **官方 DuoAttention 基线打通**: 独立 env(transformers 4.44.2,
   flash-attn 2.8.1)跑官方 duo_attn.patch;官方代码与 transformers
   ≥4.5x 不兼容(属性漂移 + TP), 需 4.44 + 单卡适配。
3. 实验协议必须逐字段对齐(模板/max_gen/post_process/样本数),
   否则 F1 差 100 倍(旧协议 0.14 vs 官方协议 31.5)。
