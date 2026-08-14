"""实验 B:NIAH 损失边界网格——DuoKV 在 ratio × depth 下的找回率。

主张 4 补充:压缩的"损失边界"(什么时候会坏)。
- 长度 8K(comp 窗口 384 只覆盖 4.7%,中间 95% 靠 full heads)
- ratio 0.25/0.5/0.75 × depth 0.2/0.5/0.8 × 2 seeds = 18
- FullKV 同配置 sanity(应全中)
"""
import json
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
    rng = np.random.default_rng(seed)
    magic = int(rng.integers(100000, 999999))
    ids = []
    filler_ids = tok(FILLER, add_special_tokens=False)["input_ids"]
    while len(ids) < target_tokens:
        ids.extend(filler_ids)
    ids = ids[:target_tokens]
    needle_ids = tok(NEEDLE.format(magic), add_special_tokens=False)["input_ids"]
    pos = int(target_tokens * depth_ratio)
    ids = ids[:pos] + needle_ids + ids[pos:]
    return tok.decode(ids, skip_special_tokens=True), magic


def gen(port, text, n=16):
    q = text + "\n\nWhat is the special magic number mentioned in the text? Answer with the number only."
    payload = {"text": q, "sampling_params": {"max_new_tokens": n, "temperature": 0.0}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/generate",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as resp:
        return json.loads(resp.read().decode())["text"]


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        "/home/lixinze/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct"
    )
    PORTS = {"fullkv": 30060, "duo_0.25": 30062, "duo_0.5": 30061, "duo_0.75": 30063}

    rows = []
    for depth in (0.2, 0.5, 0.8):
        for seed in range(2):
            text, magic = build_prompt(8192, depth, seed, tok)
            hits = {}
            for name, port in PORTS.items():
                out = gen(port, text)
                hits[name] = str(magic) in out.replace(",", "")
            rows.append({"depth": depth, "seed": seed, "magic": magic, **hits})
            print(f"depth={depth} seed={seed}: " +
                  " ".join(f"{k}={v}" for k, v in hits.items()))

    with open("/home/lixinze/HeadKV-SGLang/artifacts/expB_loss_boundary.json", "w") as f:
        json.dump(rows, f, indent=2)

    print("\n=== 汇总(hit/total)===")
    for name in PORTS:
        n = sum(1 for r in rows if r[name])
        print(f"{name}: {n}/{len(rows)}")
    print("\n=== 按 depth ===")
    for depth in (0.2, 0.5, 0.8):
        sub = [r for r in rows if r["depth"] == depth]
        line = f"  depth={depth}: " + " ".join(
            f"{k}={sum(1 for r in sub if r[k])}/2" for k in PORTS)
        print(line)
    print("=== 按 ratio(duo)===")
    for ratio in ("0.25", "0.5", "0.75"):
        sub = [r for r in rows]
        n = sum(1 for r in sub if r[f"duo_{ratio}"])
        print(f"  ratio {ratio}: {n}/6")


if __name__ == "__main__":
    main()
