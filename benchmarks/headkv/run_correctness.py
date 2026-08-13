"""E2E correctness:FullKV vs DuoKV 同一 prompt 生成对比(Phase 4.2)。

用法:
    python run_correctness.py --fullkv-port 30000 --duokv-port 30001 \
        --prompts ../../artifacts/prompts_correctness.jsonl \
        --max-new-tokens 64 --out ../../artifacts/correctness_e2e.json
"""
import argparse
import json
import time
import urllib.request


def generate(port, text, max_new_tokens, timeout=300):
    payload = {
        "text": text,
        "sampling_params": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.0,
        },
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())["text"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fullkv-port", type=int, required=True)
    ap.add_argument("--duokv-port", type=int, required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.prompts) as f:
        prompts = [json.loads(l) for l in f if l.strip()]

    results = []
    n_exact = 0
    for p in prompts:
        pid, text = p["id"], p["prompt"]
        t0 = time.time()
        out_f = generate(args.fullkv_port, text, args.max_new_tokens)
        out_d = generate(args.duokv_port, text, args.max_new_tokens)
        dt = time.time() - t0
        exact = (out_f == out_d)
        n_exact += int(exact)
        results.append({
            "id": pid, "type": p.get("type"),
            "fullkv": out_f, "duokv": out_d,
            "exact_match": exact,
            "elapsed_s": round(dt, 2),
        })
        print(f"[{pid}] exact={exact} fullkv={out_f[:60]!r} duokv={out_d[:60]!r}")

    summary = {
        "total": len(results),
        "exact_match": n_exact,
        "exact_ratio": round(n_exact / len(results), 4),
        "results": results,
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nexact match: {n_exact}/{len(results)} = {summary['exact_ratio']}")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
