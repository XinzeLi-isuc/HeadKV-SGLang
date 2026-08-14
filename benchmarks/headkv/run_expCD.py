"""实验 C+D:32K/64K 服务能力 + CG 下并发正式数据。

C:32K/64K 单请求冒烟(FullKV vs DuoKV,证明长上下文服务能力)
D:CG 下 8K×bs 并发时延对比(FullKV vs DuoKV 0.5)
"""
import concurrent.futures
import json
import time
import urllib.request

FILLER = "The great river flowed through the valley, carrying stories of generations "


def build_prompt(n_tokens):
    return (FILLER * (n_tokens // 12 + 1))[: n_tokens * 3]


def gen(port, text, n=16, timeout=1800):
    t0 = time.time()
    payload = {"text": text, "sampling_params": {"max_new_tokens": n, "temperature": 0.0}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/generate",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        json.loads(resp.read().decode())
    return time.time() - t0


def main():
    out = {}

    # ---- 实验 C:32K/64K 单请求 ----
    print("=== Exp C:长上下文单请求 ===")
    for L in (32768, 65536):
        txt = build_prompt(L)
        row = {"len": L}
        for name, port in (("fullkv", 30060), ("duokv", 30061)):
            try:
                t = gen(port, txt)
                row[name] = round(t, 2)
                print(f"  {L//1024}K {name}: ok {t:.2f}s")
            except Exception as e:  # noqa: BLE001
                row[name] = f"FAIL {type(e).__name__}"
                print(f"  {L//1024}K {name}: FAIL {e}")
        out[f"longctx_{L}"] = row

    # ---- 实验 D:CG 并发 8K × bs ----
    print("\n=== Exp D:CG 并发 8K × bs(16 decode)===")
    txt8k = build_prompt(8192)
    for bs in (16, 32):
        row = {"bs": bs}
        for name, port in (("fullkv", 30060), ("duokv", 30061)):
            t0 = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=bs) as ex:
                ts = list(ex.map(lambda _: gen(port, txt8k, 16), range(bs)))
            ts.sort()
            wall = time.time() - t0
            row[name] = {"wall_s": round(wall, 1), "median_s": round(ts[bs // 2], 2),
                         "max_s": round(ts[-1], 2)}
            print(f"  bs={bs} {name}: wall {wall:.1f}s median {ts[bs//2]:.2f}s "
                  f"max {ts[-1]:.2f}s")
        out[f"cg_conc_{bs}"] = row

    with open("/home/lixinze/HeadKV-SGLang/artifacts/expCD.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved -> artifacts/expCD.json")


if __name__ == "__main__":
    main()
