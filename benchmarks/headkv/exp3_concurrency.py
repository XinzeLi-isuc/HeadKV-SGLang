"""Experiment 3 (Batch/Concurrency): 8K/16K, BS 扫描至 FullKV 无法继续。

关键点: total KV = ctx_len x BS 跨过 FullKV T0=204824 时 FullKV 开始
排队/失败, HeadKV Tf=397360 继续承载 —— Head-wise KV compression 的
系统价值点。

判定"不能继续": fail 比例 > 30% 或 median e2e 超过单请求基线 10 倍
(排队爆炸)。每个 (ctx, bs) 先打 FullKV 再打 HeadKV。

用法: python exp3_concurrency.py <fullkv_port> <headkv_port>
"""
import concurrent.futures
import json
import sys
import time
import urllib.request
import urllib.error

FILLER = "The great river flowed through the valley, carrying stories of generations "
# 8K 扫到 96(FullKV 排队爆炸点), 16K 到 64(每请求排队已分钟级)
SWEEP = {8192: [1, 2, 4, 8, 16, 24, 32, 48, 64, 96],
         16384: [1, 2, 4, 8, 16, 24, 32, 48, 64]}
MAX_NEW = 16
TIMEOUT = 600  # 单请求超时(排队爆炸判定)


def build_prompt(n_tokens):
    return (FILLER * (n_tokens // 12 + 1))[: n_tokens * 3]


def gen_once(port, text, max_new_tokens=MAX_NEW):
    t0 = time.time()
    payload = {"text": text, "sampling_params": {"max_new_tokens": max_new_tokens,
                                                 "temperature": 0.0}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/generate",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            json.loads(resp.read().decode())
        return time.time() - t0
    except Exception:  # noqa: BLE001  (超时/连接拒绝/5xx)
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
    port_f, port_d = int(sys.argv[1]), int(sys.argv[2])
    out = []
    for ctx, bs_list in SWEEP.items():
        text = build_prompt(ctx)
        print(f"=== ctx={ctx} (total KV @BS32 = {ctx * 32} tokens; "
              f"FullKV T0=204824, HeadKV Tf=397360) ===")
        for bs in bs_list:
            r_f = sweep(port_f, text, bs)
            r_d = sweep(port_d, text, bs)
            row = {"ctx_len": ctx, "bs": bs, "fullkv": r_f, "headkv": r_d}
            out.append(row)
            flag = ""
            if r_f["fail"] > bs * 0.3:
                flag = "  <-- FullKV 不能继续"
            elif r_f.get("median_s", 0) and r_f["median_s"] > 60:
                flag = "  <-- FullKV 延迟不可用(P50>60s)"
            print(f"  bs={bs:2d}: fullkv ok={r_f['ok']}/{bs} "
                  f"med={r_f.get('median_s','-')}s | "
                  f"headkv ok={r_d['ok']}/{bs} med={r_d.get('median_s','-')}s{flag}")
    path = "/home/lixinze/HeadKV-SGLang/artifacts/exp3_concurrency.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
