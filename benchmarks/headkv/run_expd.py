"""Experiment D 简化版:连续混合负载流,统计 E2E 延迟分位与吞吐(Phase 7)。

memory-light:短 prompt 低并发;memory-bound:长 prompt 高并发。
每个负载跑 60s 连续请求流。
"""
import concurrent.futures
import json
import queue
import threading
import time
import urllib.request

FILLER = "The great river flowed through the valley, carrying stories of generations "


def build_prompt(n_tokens):
    return (FILLER * (n_tokens // 12 + 1))[: n_tokens * 3]


def gen_once(port, text, max_new_tokens=32):
    t0 = time.time()
    payload = {"text": text, "sampling_params": {"max_new_tokens": max_new_tokens,
                                                 "temperature": 0.0}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/generate",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        json.loads(resp.read().decode())
    return time.time() - t0


def load_stream(port, prompt_pool, workers, duration_s, out_q):
    """持续投递请求 duration_s 秒,记录完成时延。"""
    stop = time.time() + duration_s
    idx = 0

    def worker():
        nonlocal idx
        while True:
            if time.time() > stop:
                return
            prompt, dec = prompt_pool[idx % len(prompt_pool)]
            idx += 1
            try:
                t = gen_once(port, prompt, dec)
                out_q.put(t)
            except Exception:  # noqa: BLE001
                out_q.put(None)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(lambda _: worker(), range(workers)))


def run(port, workload, duration_s=60):
    q = queue.Queue()
    load_stream(port, workload["prompts"], workload["workers"], duration_s, q)
    times = [t for t in list(q.queue) if t is not None]
    times.sort()
    n = len(times)
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "req_per_s": round(n / duration_s, 2),
        "p50_s": round(times[n // 2], 2),
        "p95_s": round(times[int(n * 0.95) - 1], 2),
        "max_s": round(times[-1], 2),
    }


def main():
    short_pool = [(build_prompt(512), 32) for _ in range(4)]
    long_pool = [(build_prompt(8192), 64) for _ in range(4)]
    workloads = {
        "memory_light": {"prompts": short_pool, "workers": 8},
        "memory_bound": {"prompts": long_pool, "workers": 16},
    }
    out = {}
    for wname, w in workloads.items():
        print(f"=== {wname}(workers={w['workers']}, 60s)===")
        r_f = run(30020, w)
        print(f"  fullkv: {r_f}")
        r_d = run(30021, w)
        print(f"  duokv : {r_d}")
        out[wname] = {"fullkv": r_f, "duokv": r_d}
    with open("/home/lixinze/HeadKV-SGLang/artifacts/exp_d_online.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
