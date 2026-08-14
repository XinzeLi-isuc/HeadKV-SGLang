"""S1 质量对照:Duo(ratio 0.5)vs RLKV(sparsity 0.5)的 NIAH + LongBench。

同 effective ratio 0.5(F=128/C=128),各自官方 window。
"""
import argparse
import concurrent.futures
import json
import re
import time
import urllib.request

from datasets import load_dataset

FILLER = (
    "The history of the ancient kingdom spans many centuries, with its capital "
    "city serving as a crossroads of trade, culture, and scholarship. Scholars "
    "from distant lands gathered in its libraries, documenting laws, poetry, "
    "and the movements of the stars. Each generation added new chapters to the "
    "chronicle, preserving knowledge for those who would come after. "
)
NEEDLE = "The special magic number is {}. Remember it."


def build_niah(target_tokens, depth_ratio, seed, tok):
    import numpy as np
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


def normalize(s):
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(re.sub(r"[^a-z0-9 ]", "", s).split())


def f1(pred, gold):
    p, g = set(normalize(pred).split()), set(normalize(gold).split())
    if not p or not g:
        return 0.0
    inter = p & g
    prec = len(inter) / len(p)
    rec = len(inter) / len(g)
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duo-port", type=int, required=True)
    ap.add_argument("--rlkv-port", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        "/home/lixinze/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct"
    )
    out = {}

    # ---- NIAH 4K(3 depth × 3 seed = 9)----
    niah_rows = []
    for d in (0.2, 0.5, 0.8):
        for seed in range(3):
            text, magic = build_niah(4096, d, seed, tok)
            q = text + "\n\nWhat is the special magic number mentioned in the text? Answer with the number only."
            h_d = str(magic) in gen(args.duo_port, q, 16).replace(",", "")
            h_r = str(magic) in gen(args.rlkv_port, q, 16).replace(",", "")
            niah_rows.append({"depth": d, "seed": seed, "magic": magic,
                              "hit_duo": h_d, "hit_rlkv": h_r})
    out["niah_4k"] = {
        "n": len(niah_rows),
        "hit_duo": sum(1 for r in niah_rows if r["hit_duo"]),
        "hit_rlkv": sum(1 for r in niah_rows if r["hit_rlkv"]),
        "rows": niah_rows,
    }
    print(f"NIAH 4K: duo {out['niah_4k']['hit_duo']}/{len(niah_rows)} "
          f"rlkv {out['niah_4k']['hit_rlkv']}/{len(niah_rows)}")

    # ---- LongBench 2 子任务(各 30)----
    for task in ("narrativeqa", "2wikimqa"):
        ds = load_dataset("THUDM/LongBench", task, split="test", trust_remote_code=True)
        rows = []
        for i in range(min(30, len(ds))):
            ex = ds[i]
            context = ex["context"][:16384]
            prompt = f"Context:\n{context}\n\nQuestion: {ex['input']}\n\nAnswer:"
            gold = ex["answers"][0]

            def eval_one(p):
                return gen(p, prompt)

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex_:
                pred_d, pred_r = list(ex_.map(eval_one, [args.duo_port, args.rlkv_port]))
            rows.append({"f1_duo": f1(pred_d, gold), "f1_rlkv": f1(pred_r, gold)})
        f_d = sum(r["f1_duo"] for r in rows) / len(rows)
        f_r = sum(r["f1_rlkv"] for r in rows) / len(rows)
        out[task] = {"n": len(rows), "f1_duo": round(f_d, 4),
                     "f1_rlkv": round(f_r, 4), "rows": rows}
        print(f"{task}: duo F1={f_d:.4f} rlkv F1={f_r:.4f}")

    with open(args.out, "w") as f:
        json.dump({k: (v if k in ("niah_4k",) else {kk: vv for kk, vv in v.items() if kk != "rows"})
                   for k, v in out.items()} | {f"{k}_rows": v["rows"] for k, v in out.items()},
                  f, indent=2, ensure_ascii=False)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
