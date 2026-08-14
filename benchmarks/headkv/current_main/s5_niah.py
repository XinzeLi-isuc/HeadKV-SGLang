"""S5: current main 质量 —— NIAH 4K(FullKV-cg 30090 vs DuoKV-cg 30091)。

口径对齐 S1: 4K 上下文, 3 depth(0.2/0.5/0.8) x 3 seeds = 9 题,
temperature=0, 输出含 magic number 判 hit。
"""
import json
import urllib.request

import numpy as np
from transformers import AutoTokenizer

TOK = "/home/lixinze/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct"
OUT = "/home/lixinze/HeadKV-SGLang/artifacts/s5_niah_main.json"

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


def gen(port, text, max_new=64):
    payload = {"text": text, "sampling_params": {"max_new_tokens": max_new,
                                                 "temperature": 0.0}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/generate",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as resp:
        return json.loads(resp.read().decode())["text"]


def main():
    tok = AutoTokenizer.from_pretrained(TOK)
    rows, hits = [], {"fullkv": [], "duokv": []}
    for depth in DEPTHS:
        for seed in SEEDS:
            text, magic = build_niah(TARGET, depth, seed, tok)
            # 对齐 v0.5.2 S1: 带 instruction, max_new=16, 判定去逗号
            q = text + ("\n\nWhat is the special magic number mentioned in "
                        "the text? Answer with the number only.")
            row = {"depth": depth, "seed": seed, "magic": magic}
            for name, port in [("fullkv", 30090), ("duokv", 30091)]:
                out = gen(port, q, max_new=16)
                hit = str(magic) in out.replace(",", "")
                hits[name].append(hit)
                row[name + "_hit"] = hit
                row[name + "_out"] = out[:60]
                print(f"[d={depth} s={seed}] {name}: hit={hit} out={out[:50]!r}")
            rows.append(row)
    print(f"\n=== current main NIAH 4K({len(rows)} 题)===")
    for name in ["fullkv", "duokv"]:
        n = sum(hits[name])
        print(f"{name}: {n}/{len(hits[name])} ({n / len(hits[name]) * 100:.1f}%)")
    with open(OUT, "w") as f:
        json.dump({"n": len(rows), "target_tokens": TARGET,
                   "hits": {k: sum(v) for k, v in hits.items()}, "rows": rows},
                  f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
