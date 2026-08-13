"""Mini-NIAH:长文本中间插 needle,FullKV vs DuoKV vs Official Duo 找回率。

Phase 4.2 小样本(每长度 5 条,多 depth)。完整 NIAH 在 Phase 7 Experiment E。
"""
import json
import os
import urllib.request

import numpy as np

FILLER = (
    "The history of the ancient kingdom spans many centuries, with its capital "
    "city serving as a crossroads of trade, culture, and scholarship. Scholars "
    "from distant lands gathered in its libraries, documenting laws, poetry, "
    "and the movements of the stars. Each generation added new chapters to the "
    "chronicle, preserving knowledge for those who would come after. "
)
NEEDLE = "The special magic number is {}. Remember it."


def build_prompt(target_tokens, depth_ratio, seed, tok):
    """构造 filler + needle 的长 prompt。depth_ratio 为 needle 相对位置。"""
    rng = np.random.default_rng(seed)
    magic = rng.integers(100000, 999999)
    ids_all = []
    # 填 filler 到 target_tokens
    filler_ids = tok(FILLER, add_special_tokens=False)["input_ids"]
    while len(ids_all) < target_tokens:
        ids_all.extend(filler_ids)
    ids_all = ids_all[: target_tokens]
    # 插入 needle
    needle_ids = tok(NEEDLE.format(magic), add_special_tokens=False)["input_ids"]
    pos = int(target_tokens * depth_ratio)
    ids_all = ids_all[:pos] + needle_ids + ids_all[pos:]
    text = tok.decode(ids_all, skip_special_tokens=True)
    return text, int(magic)


def ask(port, text):
    q = text + "\n\nWhat is the special magic number mentioned in the text? "
    q += "Answer with the number only."
    payload = {"text": q, "sampling_params": {"max_new_tokens": 16, "temperature": 0.0}}
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/generate",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode())["text"]


def main():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        "/home/lixinze/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct"
    )
    ports = {"fullkv_triton": 30003, "duokv": 30001}
    lens = [4096]
    depths = [0.2, 0.5, 0.8]
    rows = []
    for L in lens:
        for d in depths:
            for seed in range(3):
                text, magic = build_prompt(L, d, seed, tok)
                outs = {k: ask(p, text) for k, p in ports.items()}
                row = {"len": L, "depth": d, "seed": seed, "magic": magic}
                for k, o in outs.items():
                    hit = str(magic) in o.replace(",", "")
                    row[f"hit_{k}"] = hit
                    row[f"out_{k}"] = o.strip()[:40]
                rows.append(row)
                print(f"L={L} d={d} seed={seed} magic={magic} "
                      f"fullkv={row['hit_fullkv_triton']} duokv={row['hit_duokv']}")

    os.makedirs("../../artifacts", exist_ok=True)
    with open("../../artifacts/niah_mini.json", "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    n = len(rows)
    for k in ("fullkv_triton", "duokv"):
        hits = sum(1 for r in rows if r[f"hit_{k}"])
        print(f"\n{k}: {hits}/{n} = {hits/n:.2f}")


if __name__ == "__main__":
    main()
