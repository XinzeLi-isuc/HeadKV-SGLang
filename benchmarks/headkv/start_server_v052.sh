#!/usr/bin/env bash
# v0.5.2 主战场 server 启动器(E2/E3/E4 实验用)
# 用法: bash start_server_v052.sh <name> <gpu> <port> <policy:fullkv|duo|rlkv> [eager]
set -e

NAME=$1; GPU=$2; PORT=$3; POLICY=$4; MODE=${5:-cg}
MODEL=/home/lixinze/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct
DUO=/home/lixinze/duo-attention-ref/attn_patterns/Meta-Llama-3.1-8B-Instruct/lr=0.02-reg=0.05-ctx=1000_128000-multi_passkey10
RLKV=/home/lixinze/rlkv/head_dist/rlkv/Llama-3.1-8B-R1/llama_lr1e-2_ep2_bs32_reg1e-3_tau0.5
LOG=/home/lixinze/HeadKV-SGLang/artifacts/exp2_${NAME}.log

COMMON="--model-path $MODEL --port $PORT --mem-fraction-static 0.85 \
  --max-running-requests 32 --disable-radix-cache \
  --triton-attention-num-kv-splits 32"

if [ "$POLICY" = "fullkv" ]; then
  EXTRA="--attention-backend triton"
elif [ "$POLICY" = "duo" ]; then
  EXTRA="--enable-headkv --headkv-policy duo --headkv-pattern-path $DUO \
    --headkv-full-head-ratio 0.5"
elif [ "$POLICY" = "rlkv" ]; then
  EXTRA="--enable-headkv --headkv-policy rlkv --headkv-pattern-path $RLKV \
    --rlkv-sparsity 0.5"
else
  echo "unknown policy: $POLICY"; exit 1
fi

if [ "$MODE" = "eager" ]; then
  EXTRA="$EXTRA --disable-cuda-graph"
fi

echo "[$NAME] GPU=$GPU PORT=$PORT POLICY=$POLICY MODE=$MODE"
CUDA_VISIBLE_DEVICES=$GPU python -m sglang.launch_server $COMMON $EXTRA 2>&1 | tee $LOG
