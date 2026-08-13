"""构造指定 token 长度的 smoke prompt(用于 FullKV / DuoKV smoke 与 correctness 基线)。

用法:
    python make_smoke_prompt.py --model MODEL_PATH --tokens 4096 --out prompt_4k.txt

用模型自带 tokenizer 精确截断到目标长度,避免手工文本长度误差。
"""
import argparse
import json
import os

import torch
from transformers import AutoTokenizer

FILLER = (
    "Attention mechanisms in large language models allow every token to attend to every "
    "previous token in the sequence. This quadratic complexity is the key bottleneck for "
    "long-context inference. Systems like PagedAttention manage the KV cache in fixed-size "
    "blocks so that memory fragmentation is reduced and batches of concurrent requests can "
    "share GPU memory efficiently. Streaming attention, on the other hand, keeps only a "
    "small sink of initial tokens together with a sliding window of recent tokens, trading "
    "full recall for constant memory. The trade-off between accuracy and memory usage is "
    "the central design question for efficient serving of long sequences. "
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model path (local) for tokenizer")
    ap.add_argument("--tokens", type=int, default=4096, help="target token count")
    ap.add_argument("--out", required=True, help="output .txt path")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    # 用足够长的种子文本一次性 encode,再精确截断
    n_rep = max(1, args.tokens // 64 + 2)
    seed = FILLER * n_rep
    ids = tok(seed, add_special_tokens=False)["input_ids"][: args.tokens]
    text = tok.decode(ids, skip_special_tokens=True)

    with open(args.out, "w") as f:
        f.write(text)

    n_final = len(tok(text, add_special_tokens=False)["input_ids"])
    print(f"target={args.tokens} actual={n_final} saved={args.out}")


if __name__ == "__main__":
    main()
