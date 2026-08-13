"""Experiment E:LongBench 2 子任务评估(Phase 7)。

narrativeqa / 2wikimqa:context + question → 生成答案 → F1(官方指标简化版)。
对比 FullKV-triton vs DuoKV。
"""
import argparse
import concurrent.futures
import json
import re
import time
import urllib.request
import urllib.error

from datasets import load_dataset


def normalize(s):
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9 ]", "", s)
    return " ".join(s.split())


def f1_score(pred, gold):
    p_tok, g_tok = set(normalize(pred).split()), set(normalize(gold).split())
    if not p_tok or not g_tok:
        return 0.0
    inter = p_tok & g_tok
    p = len(inter) / len(p_tok)
    r = len(inter) / len(g_tok)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def gen(port, text, max_new_tokens=64):
    payload = {"text": text, "sampling_params": {"max_new_tokens": max_new_tokens,
                                                 "temperature": 0.0}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/generate",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as resp:
        return json.loads(resp.read().decode())["text"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fullkv-port", type=int, required=True)
    ap.add_argument("--duokv-port", type=int, required=True)
    ap.add_argument("--max-samples", type=int, default=50)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = {"narrativeqa": None, "2wikimqa": None}
    for task in ("narrativeqa", "2wikimqa"):
        ds = load_dataset("THUDM/LongBench", task, split="test",
                          trust_remote_code=True)
        rows = []
        for i in range(min(args.max_samples, len(ds))):
            ex = ds[i]
            if task == "narrativeqa":
                context = ex["context"][:16384]
                q = ex["input"]
                gold = ex["answers"][0]
            else:  # 2wikimqa
                context = ex["context"][:16384]
                q = ex["input"]
                gold = ex["answers"][0]
            prompt = f"Context:\n{context}\n\nQuestion: {q}\n\nAnswer:"
            rows.append({"i": i, "prompt": prompt, "gold": gold})

        # 用线程池双端口交错发,避免预热偏差
        def eval_one(r):
            pred_f = gen(args.fullkv_port, r["prompt"])
            pred_d = gen(args.duokv_port, r["prompt"])
            return {
                "i": r["i"],
                "gold": r["gold"][:60],
                "pred_fullkv": pred_f.strip()[:120],
                "pred_duokv": pred_d.strip()[:120],
                "f1_fullkv": f1_score(pred_f, r["gold"]),
                "f1_duokv": f1_score(pred_d, r["gold"]),
            }

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            res = list(ex.map(eval_one, rows))
        f1_f = sum(r["f1_fullkv"] for r in res) / len(res)
        f1_d = sum(r["f1_duokv"] for r in res) / len(res)
        out[task] = {"n": len(res), "f1_fullkv": round(f1_f, 4),
                     "f1_duokv": round(f1_d, 4), "rows": res}
        print(f"{task}: fullkv F1={f1_f:.4f} duokv F1={f1_d:.4f} (n={len(res)})")

    with open(args.out, "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "rows"}
                   for k, v in out.items()} | {f"{k}_rows": v["rows"] for k, v in out.items()},
                  f, indent=2, ensure_ascii=False)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
