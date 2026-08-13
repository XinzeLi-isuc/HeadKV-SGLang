"""Exp C 聚焦版:8K prompt 并发 16/32/48,对比 E2E 时延分布。

甜蜜点:总量 ∈ (FullKV T0=204824, DuoKV Tf=385072]
  8K × 32 = 262144 → FullKV 需排队,DuoKV 全驻留
"""
import concurrent.futures
import json
import time
import urllib.request
import urllib.error

FILLER = "The great river flowed through the valley, carrying stories of generations "


def build_prompt(n_tokens):
    return (FILLER * (n_tokens // 12 + 1))[: n_tokens * 3]


def gen_once(port, text, max_new_tokens=16):
    t0 = time.time()
    payload = {"text": text, "sampling_params": {"max_new_tokens": max_new_tokens,
                                                 "temperature": 0.0}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/generate",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            json.loads(resp.read().decode())
        return time.time() - t0
    except Exception as e:  # noqa: BLE001
        return None


def sweep(port, text, bs):
    with concurrent.futures.ThreadPoolExecutor(max_workers=bs) as ex:
        times = list(ex.map(lambda _: gen_once(port, text), range(bs)))
    ok = [t for t in times if t is not None]
    ok.sort()
    if not ok:
        return {"ok": 0, "fail": bs}
    return {"ok": len(ok), "fail": bs - len(ok),
            "median_s": round(ok[len(ok) // 2], 2),
            "p95_s": round(ok[int(len(ok) * 0.95) - 1], 2),
            "max_s": round(ok[-1], 2)}


def main():
    out = []
    txt8k = build_prompt(8192)
    print("8K prompt 并发 sweep(16 decode tokens/req)")
    for bs in (16, 32, 48):
        r_f = sweep(30020, txt8k, bs)
        r_d = sweep(30021, txt8k, bs)
        row = {"bs": bs, "fullkv": r_f, "duokv": r_d}
        out.append(row)
        print(f"  bs={bs}: fullkv={r_f}")
        print(f"          duokv={r_d}")
    with open("/home/lixinze/HeadKV-SGLang/artifacts/exp_c_concurrency.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
