"""Experiment B/C:context length 单请求指标 + concurrency sweep(Phase 7)。

B:4K/8K/16K 单请求,记录 E2E 时间(含 prefill+decode)、输出 tok/s
C:固定 8K/16K prompt,并发 1..N 递增,统计成功率(FullKV vs DuoKV 峰值)

用法:
    python run_experiments.py --fullkv-port 30000 --duokv-port 30001 \
        --out ../../artifacts/exp_bc.json
"""
import argparse
import concurrent.futures
import json
import time
import urllib.request
import urllib.error

FILLER = "The great river flowed through the valley, carrying stories of generations "


def build_prompt(n_tokens):
    return (FILLER * (n_tokens // 12 + 1))[: n_tokens * 3]


def gen_once(port, text, max_new_tokens, timeout=600):
    t0 = time.time()
    payload = {"text": text, "sampling_params": {"max_new_tokens": max_new_tokens,
                                                 "temperature": 0.0}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/generate",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            out = json.loads(resp.read().decode())["text"]
        return {"ok": True, "elapsed": time.time() - t0, "out": out}
    except urllib.error.HTTPError as e:
        return {"ok": False, "elapsed": time.time() - t0, "err": f"http_{e.code}"}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "elapsed": time.time() - t0, "err": str(e)[:80]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fullkv-port", type=int, required=True)
    ap.add_argument("--duokv-port", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    results = {"exp_b": [], "exp_c": []}

    # ---- Experiment B:context length ----
    print("=== Exp B: context length ===")
    for L in (4096, 8192, 16384):
        txt = build_prompt(L)
        row = {"len": L}
        for name, port in (("fullkv", args.fullkv_port), ("duokv", args.duokv_port)):
            times = []
            for _ in range(3):
                r = gen_once(port, txt, 32)
                if r["ok"]:
                    times.append(r["elapsed"])
            if times:
                row[name] = {"n": len(times),
                             "median_s": round(sorted(times)[len(times) // 2], 3),
                             "tok_per_s": round(32 / (sum(times) / len(times)), 2)}
            else:
                row[name] = {"n": 0, "err": "all_failed"}
        results["exp_b"].append(row)
        print(f"  L={L}: {row}")

    # ---- Experiment C:concurrency sweep ----
    print("=== Exp C: concurrency sweep ===")
    for L in (8192, 16384):
        txt = build_prompt(L)
        for bs in (1, 2, 4, 8, 16, 32, 64):
            row = {"len": L, "bs": bs}
            for name, port in (("fullkv", args.fullkv_port), ("duokv", args.duokv_port)):
                jobs = [{"port": port, "text": txt, "n": 16} for _ in range(bs)]
                with concurrent.futures.ThreadPoolExecutor(max_workers=bs) as ex:
                    rs = list(ex.map(lambda j: gen_once(j["port"], j["text"], j["n"]), jobs))
                ok = sum(1 for r in rs if r["ok"])
                row[name] = {"ok": ok, "fail": bs - ok}
            results["exp_c"].append(row)
            print(f"  L={L} bs={bs}: fullkv={row['fullkv']} duokv={row['duokv']}")
            if row["fullkv"]["ok"] == 0 and row["duokv"]["ok"] == 0:
                break  # 双方都打满

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
