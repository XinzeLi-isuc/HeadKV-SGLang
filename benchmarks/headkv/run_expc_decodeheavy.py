"""Exp C decode-heavy:4K prompt × 64 并发 × 128 decode。

总量 4K×64 = 262K tokens:
  FullKV T0=204824 < 262K → decode 阶段池满,请求排队(吞吐受限)
  DuoKV  Tf=385072 > 262K → 全驻留,无排队
对比:总完成时间 / 吞吐
"""
import concurrent.futures
import json
import time
import urllib.request

FILLER = "The great river flowed through the valley, carrying stories of generations "


def build_prompt(n_tokens):
    return (FILLER * (n_tokens // 12 + 1))[: n_tokens * 3]


def gen_once(port, text, max_new_tokens=128):
    t0 = time.time()
    payload = {"text": text, "sampling_params": {"max_new_tokens": max_new_tokens,
                                                 "temperature": 0.0}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/generate",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=1800) as resp:
            json.loads(resp.read().decode())
        return time.time() - t0
    except Exception as e:  # noqa: BLE001
        return None


def run(port, text, bs):
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=bs) as ex:
        times = list(ex.map(lambda _: gen_once(port, text), range(bs)))
    wall = time.time() - t0
    ok = [t for t in times if t is not None]
    ok.sort()
    total_tokens = bs * 128
    return {"ok": len(ok), "fail": bs - len(ok), "wall_s": round(wall, 1),
            "gen_tok_s": round(total_tokens / wall, 1),
            "median_s": round(ok[len(ok) // 2], 1),
            "max_s": round(ok[-1], 1)}


def main():
    txt = build_prompt(4096)
    bs = 64
    print(f"4K × {bs} 并发 × 128 decode(总量 {4 * bs}K tokens)")
    r_f = run(30020, txt, bs)
    print(f"  fullkv: {r_f}")
    r_d = run(30021, txt, bs)
    print(f"  duokv : {r_d}")
    with open("/home/lixinze/HeadKV-SGLang/artifacts/exp_c_decodeheavy.json", "w") as f:
        json.dump({"fullkv": r_f, "duokv": r_d}, f, indent=2)


if __name__ == "__main__":
    main()
