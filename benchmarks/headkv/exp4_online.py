"""Experiment 4 (Online Serving): FullKV vs HeadKV 连续负载, TTFT/TPOT 分解。

workload 对齐 exp_d:
- memory_light: 短 prompt(512 tok) x 8 workers, 32 gen tok
- memory_bound: 长 prompt(16K tok) x 8 workers, 128 gen tok

流式记录每请求 TTFT/TPOT/e2e, 60s 连续流, 输出:
request/s, output tok/s, P50/P95 TTFT, P50/P95 TPOT, max running requests。

用法: python exp4_online.py <fullkv_port> <headkv_port> [duration_s=60]
"""
import concurrent.futures
import json
import queue
import statistics
import sys
import time
import urllib.request

FILLER = "The great river flowed through the valley, carrying stories of generations "

WORKLOADS = {
    "memory_light": {"prompt_tok": 512, "max_new": 32, "workers": 8},
    "memory_bound": {"prompt_tok": 16384, "max_new": 128, "workers": 8},
}


def build_prompt(n_tokens):
    return (FILLER * (n_tokens // 12 + 1))[: n_tokens * 3]


def gen_stream(port, text, max_new):
    """返回 (ttft_s, e2e_s, n_tokens) 或 None(失败)。"""
    payload = {"text": text, "sampling_params": {"max_new_tokens": max_new,
                                                 "temperature": 0.0},
               "stream": True}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/generate",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    n_chunks = 0
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            for line in resp:
                line = line.decode().strip()
                if not line.startswith("data:"):
                    continue
                if ttft is None:
                    ttft = time.time() - t0
                n_chunks += 1
    except Exception:  # noqa: BLE001
        return None
    return ttft, time.time() - t0, n_chunks


def load_stream(port, prompt, max_new, workers, duration_s, out_q):
    stop = time.time() + duration_s

    def worker():
        while time.time() < stop:
            r = gen_stream(port, prompt, max_new)
            out_q.put(r)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(lambda _: worker(), range(workers)))


def run(port, workload, duration_s):
    q = queue.Queue()
    prompt = build_prompt(workload["prompt_tok"])
    load_stream(port, prompt, workload["max_new"], workload["workers"],
                duration_s, q)
    rows = [r for r in list(q.queue) if r is not None]
    if not rows:
        return {"n": 0}
    ttfts = sorted(r[0] for r in rows)
    tpots = sorted((r[1] - r[0]) / max(r[2] - 1, 1) for r in rows)
    gen_toks = sum(r[2] - 1 for r in rows)
    n = len(rows)
    return {
        "n": n,
        "req_per_s": round(n / duration_s, 2),
        "output_tok_per_s": round(gen_toks / duration_s, 1),
        "p50_ttft_s": round(ttfts[n // 2], 3),
        "p95_ttft_s": round(ttfts[int(n * 0.95) - 1], 3),
        "p50_tpot_s": round(tpots[n // 2], 4),
        "p95_tpot_s": round(tpots[int(n * 0.95) - 1], 4),
        "max_running_requests": workload["workers"],
    }


def main():
    port_f, port_d = int(sys.argv[1]), int(sys.argv[2])
    duration = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    out = {}
    for wname, wl in WORKLOADS.items():
        print(f"=== workload={wname} (prompt={wl['prompt_tok']}, "
              f"gen={wl['max_new']}, workers={wl['workers']}) ===")
        r_f = run(port_f, wl, duration)
        r_d = run(port_d, wl, duration)
        out[wname] = {"fullkv": r_f, "headkv": r_d}
        for name, r in [("fullkv", r_f), ("headkv", r_d)]:
            print(f"  {name}: req/s={r.get('req_per_s')} out_tok/s={r.get('output_tok_per_s')} "
                  f"TTFT p50/p95={r.get('p50_ttft_s')}/{r.get('p95_ttft_s')}s "
                  f"TPOT p50/p95={r.get('p50_tpot_s')}/{r.get('p95_tpot_s')}s")
    path = "/home/lixinze/HeadKV-SGLang/artifacts/exp4_online.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
