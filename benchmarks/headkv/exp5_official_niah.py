"""Experiment 5: Official DuoAttention 基线 —— NIAH 4K(magic number 协议)。

与 HeadKV 的 run_s1_quality.py 同一协议(3 depth x 3 seed, magic hit 判定),
用官方 duo-attention(duo_attn.patch)在 HF 上直接生成:
- --method full: Full Attention 基线
- --method duo_attn: Official DuoAttention(sparsity 0.5, sink 128, recent 256)

用法: python exp5_official_niah.py <full|duo_attn>
输出: artifacts/exp5_official_niah_<method>.json
"""
import argparse
import json
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/home/lixinze/duo-attention-ref")
from duo_attn.patch import enable_duo_attention_eval  # noqa: E402
from duo_attn.utils import load_attn_pattern, sparsify_attention_heads  # noqa: E402

TOK = "/home/lixinze/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct"
PATTERN = ("/home/lixinze/duo-attention-ref/attn_patterns/Meta-Llama-3.1-8B-Instruct/"
           "lr=0.02-reg=0.05-ctx=1000_128000-multi_passkey10")

FILLER = (
    "The history of the ancient kingdom spans many centuries, with its capital "
    "city serving as a crossroads of trade, culture, and scholarship. Scholars "
    "from distant lands gathered in its libraries, documenting laws, poetry, "
    "and the movements of the stars. Each generation added new chapters to the "
    "chronicle, preserving knowledge for those who would come after. "
)
NEEDLE = "The special magic number is {}. Remember it."
DEPTHS = [0.2, 0.5, 0.8]
SEEDS = [0, 1, 2]
TARGET = 4096
MAX_NEW = 16


def build_niah(target_tokens, depth_ratio, seed, tok):
    rng = np.random.default_rng(seed)
    magic = int(rng.integers(100000, 999999))
    ids_all = []
    filler_ids = tok(FILLER, add_special_tokens=False)["input_ids"]
    while len(ids_all) < target_tokens:
        ids_all.extend(filler_ids)
    ids_all = ids_all[: target_tokens]
    needle_ids = tok(NEEDLE.format(magic), add_special_tokens=False)["input_ids"]
    pos = int(target_tokens * depth_ratio)
    ids_all = ids_all[:pos] + needle_ids + ids_all[pos:]
    return tok.decode(ids_all, skip_special_tokens=True), magic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("method", choices=["full", "duo_attn"])
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(TOK)
    model = AutoModelForCausalLM.from_pretrained(
        TOK, torch_dtype=torch.bfloat16, attn_implementation="eager"
    ).eval().cuda()

    if args.method == "duo_attn":
        full_heads, sink_size, recent_size = load_attn_pattern(PATTERN)
        full_heads, sparsity = sparsify_attention_heads(full_heads, None, 0.5)
        enable_duo_attention_eval(model, full_heads, sink_size, recent_size)
        print(f"[official duo_attn] sink={sink_size} recent={recent_size} "
              f"sparsity={sparsity}")

    rows, hits = [], 0
    for depth in DEPTHS:
        for seed in SEEDS:
            text, magic = build_niah(TARGET, depth, seed, tok)
            q = text + ("\n\nWhat is the special magic number mentioned in "
                        "the text? Answer with the number only.")
            inputs = tok(q, return_tensors="pt").to("cuda")
            # 官方 patch 需 tuple cache: 手动 prefill + 逐 token argmax
            # (与官方 pred.py 同法, 不能走 model.generate)
            with torch.no_grad():
                output = model(input_ids=inputs["input_ids"],
                               past_key_values=None, use_cache=True)
                past_key_values = output.past_key_values
                pred_idx = output.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                gen_ids = [pred_idx.item()]
                for _ in range(MAX_NEW - 1):
                    outputs = model(input_ids=pred_idx,
                                    past_key_values=past_key_values,
                                    use_cache=True)
                    past_key_values = outputs.past_key_values
                    pred_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                    gen_ids.append(pred_idx.item())
                    if pred_idx.item() == tok.eos_token_id:
                        break
            out = tok.decode(gen_ids, skip_special_tokens=True)
            hit = str(magic) in out.replace(",", "")
            hits += hit
            rows.append({"depth": depth, "seed": seed, "magic": magic,
                         "hit": hit, "out": out[:40]})
            print(f"[d={depth} s={seed}] hit={hit} out={out[:40]!r}")
    print(f"\n=== Official {args.method} NIAH 4K: {hits}/{len(rows)} ===")
    path = f"/home/lixinze/HeadKV-SGLang/artifacts/exp5_official_niah_{args.method}.json"
    with open(path, "w") as f:
        json.dump({"method": args.method, "n": len(rows), "hits": hits,
                   "rows": rows}, f, indent=2, ensure_ascii=False)
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
