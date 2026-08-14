#!/usr/bin/env bash
# E5 官方 LongBench: narrativeqa + 2wikimqa x {full, duo_attn} x 30 条
# (与 HeadKV run_s1_quality.py 的 30 条协议对齐)
set -e
export PATH=/home/lixinze/miniconda3/envs/duo-official/bin:$PATH
export NO_PROXY="127.0.0.1,localhost"
cd ~/duo-attention-ref
MODEL=Meta-Llama-3.1-8B-Instruct
PATTERN=/home/lixinze/duo-attention-ref/attn_patterns/Meta-Llama-3.1-8B-Instruct/lr=0.02-reg=0.05-ctx=1000_128000-multi_passkey10

for task in narrativeqa 2wikimqa; do
  for method in full duo_attn; do
    echo "=== $task / $method ==="
    if [ "$method" = "duo_attn" ]; then
      python eval/LongBench/pred.py --model $MODEL --task $task --method duo_attn \
        --attn_load_dir $PATTERN --sink_size 128 --recent_size 256 \
        --sparsity 0.5 --limit 30 2>&1 | tail -3
    else
      python eval/LongBench/pred.py --model $MODEL --task $task --method full \
        --limit 30 2>&1 | tail -3
    fi
  done
done
echo "=== ALL DONE ==="
