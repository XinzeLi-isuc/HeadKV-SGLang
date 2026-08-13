#!/bin/bash
# Phase 5 容量采集:对不同 full_head_ratio 启动 DuoKV server,提取 max_total_num_tokens。
# 用法: bash run_capacity_sweep.sh  (GPU 0)
set -u
export PATH=/home/lixinze/miniconda3/envs/rlkv-eval/bin:$PATH
export NO_PROXY="127.0.0.1,localhost"
MODEL=~/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct
PATTERN=/home/lixinze/duo-attention-ref/attn_patterns/Meta-Llama-3.1-8B-Instruct/lr=0.02-reg=0.05-ctx=1000_128000-multi_passkey10
OUT=/home/lixinze/HeadKV-SGLang/artifacts/capacity_sweep.csv

echo "ratio,Tf,max_total_num_tokens" > "$OUT"
PORT=30010
for RATIO in 0.25 0.75; do
  LOG=/tmp/headkv_cap_${RATIO}.log
  CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
    --model-path "$MODEL" --port $PORT --mem-fraction-static 0.85 \
    --max-running-requests 32 --disable-radix-cache --disable-cuda-graph \
    --enable-headkv --headkv-policy duo --headkv-pattern-path "$PATTERN" \
    --headkv-full-head-ratio $RATIO > "$LOG" 2>&1 &
  SPID=$!
  # 等待 fired up(最多 180s)
  for i in $(seq 1 60); do
    if grep -q "The server is fired up" "$LOG" 2>/dev/null; then break; fi
    if ! kill -0 $SPID 2>/dev/null; then echo "server died (ratio=$RATIO)"; tail -5 "$LOG"; exit 1; fi
    sleep 3
  done
  TF=$(grep -oP "Tf=\d+" "$LOG" | head -1 | cut -d= -f2)
  MT=$(grep -oP "max_total_num_tokens=\d+" "$LOG" | head -1 | cut -d= -f2)
  echo "$RATIO,$TF,$MT" | tee -a "$OUT"
  # 提取 Gate 3 行存档
  grep "\[HeadKV\]" "$LOG" >> /home/lixinze/HeadKV-SGLang/artifacts/capacity_sweep_gate3.log
  kill $SPID 2>/dev/null
  wait $SPID 2>/dev/null
  PORT=$((PORT + 1))
  sleep 5
done
echo "done -> $OUT"
