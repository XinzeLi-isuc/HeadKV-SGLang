"""Experiment 2 (Context Length): 4K/8K/16K/32K x {fullkv, headkv-0.5}。

流式 /generate 记录:
- prefill latency = TTFT(单请求无排队, 首 token 到达时间)
- decode TPOT = (e2e - TTFT) / (n_tokens - 1)
- tokens/s(decode 吞吐)
- KV capacity(Tf, 从 server 启动日志/API 读)

用法: python exp2_context.py <fullkv_port> <headkv_port> [runs=3]
"""
import json
import statistics
import sys
import time
import urllib.request

FILLER = ("The great river flowed through the valley, carrying stories of "
          "generations of people who lived along its banks. Farmers worked "
          "the fertile soil, merchants traded goods from distant lands, and "
          "scholars recorded the history of their times in careful detail. ")
CTX_LENS = [4096, 8192, 16384, 32768]
MAX_NEW = 32


def build_prompt(n_tokens):
    return (FILLER * (n_tokens // 24 + 2))[: n_tokens * 4]


def gen_stream(port, text, max_new=MAX_NEW):
    """流式生成: 返回 (ttft_s, e2e_s, n_tokens)。"""
    payload = {"text": text, "sampling_params": {"max_new_tokens": max_new,
                                                 "temperature": 0.0},
               "stream": True}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/generate",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    n_chunks = 0
    with urllib.request.urlopen(req, timeout=900) as resp:
        for line in resp:
            line = line.decode().strip()
            if not line.startswith("data:"):
                continue
            if ttft is None:
                ttft = time.time() - t0
            n_chunks += 1
    e2e = time.time() - t0
    return ttft, e2e, n_chunks


def main():
    port_f, port_d = int(sys.argv[1]), int(sys.argv[2])
    runs = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    out = []
    for ctx in CTX_LENS:
        text = build_prompt(ctx)
        print(f"ctx={ctx} tokens(prompt ~{len(text)} chars)")
        for name, port in [("fullkv", port_f), ("headkv", port_d)]:
            ttfts, e2es, ntoks = [], [], []
            for _ in range(runs):
                ttft, e2e, n = gen_stream(port, text)
                ttfts.append(ttft)
                e2es.append(e2e)
                ntoks.append(n)
            m_ttft = statistics.median(ttfts)
            m_e2e = statistics.median(e2es)
            n_tok = statistics.median(ntoks)
            tpot = (m_e2e - m_ttft) / max(n_tok - 1, 1)
            tok_s = (n_tok - 1) / max(m_e2e - m_ttft, 1e-6)
            row = {"ctx_len": ctx, "system": name,
                   "prefill_latency_s": round(m_ttft, 3),
                   "decode_tpot_s": round(tpot, 4),
                   "tokens_per_s": round(tok_s, 2),
                   "e2e_s": round(m_e2e, 3), "gen_tokens": n_tok}
            out.append(row)
            print(f"  {name}: ttft={m_ttft:.3f}s tpot={tpot:.4f}s "
                  f"tok/s={tok_s:.1f} e2e={m_e2e:.3f}s")
    path = "/home/lixinze/HeadKV-SGLang/results/exp2_context.csv"
    with open(path, "w", newline="") as f:
        w = json  # placeholder
    import csv
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
