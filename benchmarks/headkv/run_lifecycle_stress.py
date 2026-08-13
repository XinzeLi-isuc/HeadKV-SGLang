"""Phase 6:连续批处理压测——混合长度请求循环,验证 allocator 生命周期。

设计(对应计划书 Phase 6 Gate 6):
- 2K/4K/8K/16K 混合 prompt × 不同 decode 长度
- 并发池(模拟 max_running_requests 满后排队)
- 循环 N 轮:enter → extend → decode → finish → free → slot reuse
- 校验:全部 HTTP 200、输出非空、无异常;第二轮复用相同 slot 池
- server 侧:进程存活 = allocator 未崩溃;末轮请求正常 = 无泄漏
  (若 full/comp 池泄漏,后续请求会 OOM/fail)
"""
import argparse
import concurrent.futures
import json
import random
import time
import urllib.request
import urllib.error

FILLERS = {
    2048: "The journey of a thousand miles begins with a single step. Each stride " * 40,
    4096: "The history of the ancient kingdom spans many centuries, with its capital " * 64,
    8192: "In the quiet hours of the morning, the library opened its doors to scholars " * 128,
    16384: "The great river flowed through the valley, carrying stories of generations " * 256,
}


def build_prompt(n_tokens):
    # 粗构造:直接按字符数截断(filler 足够长)
    txt = FILLERS.get(n_tokens) or (FILLERS[4096] * (n_tokens // 4096 + 1))
    return txt[: n_tokens * 3]  # 约 3 字符/token


def generate(port, text, max_new_tokens, timeout=300):
    payload = {
        "text": text,
        "sampling_params": {"max_new_tokens": max_new_tokens, "temperature": 0.0},
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
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--rounds", type=int, default=1000, help="总请求数")
    ap.add_argument("--workers", type=int, default=8, help="并发度")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = random.Random(42)
    # 请求池:混合长度(权重偏向短)
    pool = []
    for i in range(args.rounds):
        L = rng.choices([2048, 4096, 8192, 16384], weights=[4, 3, 2, 1])[0]
        dec = rng.choices([16, 32, 64], weights=[3, 2, 1])[0]
        pool.append({"id": i, "prompt": build_prompt(L), "dec": dec})

    t0 = time.time()
    ok = 0
    failures = []

    def do_one(job):
        try:
            out = generate(args.port, job["prompt"], job["dec"])
            if not out.strip():
                return job["id"], "empty_output", ""
            return job["id"], None, out
        except urllib.error.HTTPError as e:
            return job["id"], f"http_{e.code}", ""
        except Exception as e:  # noqa: BLE001
            return job["id"], f"{type(e).__name__}: {e}", ""

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for jid, err, out in ex.map(do_one, pool):
            if err is None:
                ok += 1
            else:
                failures.append({"id": jid, "err": err})

    dt = time.time() - t0
    summary = {
        "total": args.rounds,
        "ok": ok,
        "failures": failures[:20],
        "n_failures": len(failures),
        "elapsed_s": round(dt, 1),
        "req_per_s": round(args.rounds / dt, 2),
        "workers": args.workers,
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"ok={ok}/{args.rounds} failures={len(failures)} "
          f"elapsed={dt:.1f}s ({summary['req_per_s']}/s)")
    if failures:
        print("first failures:", failures[:5])


if __name__ == "__main__":
    main()
