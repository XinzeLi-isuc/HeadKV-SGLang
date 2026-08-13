# HeadKV-SGLang 环境冻结记录 (env.md)

> 每轮实验前重新执行并核对;实验期间禁止 git pull / 改依赖。

## 冻结基线 (2026-08-13 20:07)

```text
git commit : 973b5e41782be7c49978d9c01126c25708d047dc
             (Kurt232/rlkv-sglang-v0.5.2, HEAD 与计划书基线一致)
python     : 3.10.20 (conda env: rlkv-eval, 零改动直接可用)
torch      : 2.8.0+cu128 (CUDA 12.8)
triton     : 3.4.0
flashinfer : 0.3.1
numpy      : 2.2.6
transformers: 4.56.1
modelscope : 1.39.1
sglang     : 0.5.2 (editable -> ~/rlkv/sglang/python)
sitecustomize: rlkv-eval 内已装(禁 SSL 校验,HF/代理可用)
```

## 为什么零改动可用(核查证据)

1. `rlkv-eval` env 已 editable 安装本 fork:`import sglang` → `~/rlkv/sglang/python/sglang/__init__.py`
2. HeadRealloc 三组件 + Scheduler import 全部通过(2026-08-13 20:07 实测)
3. flashinfer 0.3.1 与 SGLang v0.5.2 官方匹配;triton 3.4.0 满足 fork backend 需求
4. 主模型已在本地 modelscope 缓存:
   `/home/lixinze/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3___1-8B-Instruct`
   (软链 `Meta-Llama-3.1-8B-Instruct` → 该目录,30G 完整权重,bf16)
   结构:32 layers / 32 Q heads / 8 KV heads (GQA) / hidden 4096 / max_pos 131072
5. DuoAttention pattern 已在本地:
   `/home/lixinze/duo-attention-ref/attn_patterns/Meta-Llama-3.1-8B-Instruct/
    lr=0.02-reg=0.05-ctx=1000_128000-multi_passkey10/`
   full_attention_heads.tsv: 32 行 × 8 列(与模型 GQA 完全匹配)
   config.json: sink_size=128, recent_size=256, threshold=0.5, max_length=128000
6. GPU 0/1 (A6000 48GB) 空闲;GPU 2/3 被他人进程占用,实验固定用 0/1

## 每轮实验前置检查命令

```bash
cd ~/rlkv/sglang && git rev-parse HEAD
~/miniconda3/envs/rlkv-eval/bin/python -V
~/miniconda3/envs/rlkv-eval/bin/python -c "import torch; print(torch.__version__, torch.version.cuda)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
```

## 运行入口(Phase 0 smoke,已实测通过 2026-08-13 21:32)

**两个必须的环境修正(缺一不可):**
1. `PATH` 必须包含 `~/miniconda3/envs/rlkv-eval/bin`(flashinfer JIT 编译需要 ninja,
   直接用 python 绝对路径调用时 PATH 缺失 → FileNotFoundError: ninja)
2. `NO_PROXY="127.0.0.1,localhost"`(本机 http_proxy=127.0.0.1:7884 会劫持指向
   本服务器的请求,内置 warmup 用 requests 走代理 → 502 warmup 失败)

```bash
export PATH=/home/lixinze/miniconda3/envs/rlkv-eval/bin:$PATH
export NO_PROXY="127.0.0.1,localhost"
# FullKV smoke (GPU 0)
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
  --model-path ~/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct \
  --port 30000 --mem-fraction-static 0.85 --max-running-requests 32 \
  --disable-radix-cache --disable-cuda-graph --attention-backend flashinfer
```

实测结果(2026-08-13 21:32):
- server 就绪标志:"The server is fired up and ready to roll!"(warmup 首请求含
  flashinfer JIT 编译约 39s)
- 4K prompt → 32 tokens,elapsed 1.44s,输出正确(artifacts/fullkv_smoke_resp.json)
- **T0 基线注意**:干净 GPU 0 上 max_total_num_tokens=204824;若 GPU 有残留进程,
  会缩水(实测 169437)。正式容量实验前必须 nvidia-smi 确认 0 MiB 再启动。

## 未决项(不阻塞 Phase 0)

- dtype: 模型原生 bf16,FullKV/DuoKV 同源即可(倾向 bf16 对齐权重,避免额外转换)
- `--mem-fraction-static` 默认 0.9,smoke 用 0.85 留安全边际;正式容量实验统一

## 工具补充记录

- 2026-08-13:rlkv-eval 补装 pytest 9.1.1(开发测试工具,非运行时依赖;
  其他 env 均有 pytest,仅 rlkv-eval 缺失)
