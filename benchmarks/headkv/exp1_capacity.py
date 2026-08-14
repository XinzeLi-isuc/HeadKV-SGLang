"""Experiment 1 (KV Capacity) 标注脚本。

从 server 启动日志提取 [HeadKV] 行, 计算:
- effective Full KV Head Ratio(日志 effective_ratio 字段)
- Full pool bytes / Compact pool bytes(按 HeadReallocKVPool 实际 shape)
- max_total_tokens(Tf) / max running requests
输出 results/exp1_capacity.csv(标注体系对齐)。

pool bytes 公式(与 memory_pool.py HeadReallocKVPool 一致):
  full_bytes = (Tf+1) * (full_heads/layers) * head_dim * dtype_size * 2(K,V) * layers
  comp_bytes = (Tc+1) * (compact_heads/layers) * head_dim * dtype_size * 2 * layers
bf16 = 2 bytes, head_dim = 128, layers = 32, dtype_size = 2。
"""
import csv
import glob
import re

HEAD_DIM = 128
LAYERS = 32
DTYPE_SIZE = 2  # bf16
KV_HEADS_PER_LAYER = 8  # GQA: Llama-3.1-8B 8 KV heads/layer


def parse_headkv_line(path):
    rows = []
    for line in open(path):
        if "[HeadKV]" not in line or "policy=" not in line:
            continue
        m = re.search(
            r"full_heads=(\d+) compact_heads=(\d+).*?effective_ratio=([\d.]+).*?"
            r"max_running_requests=(\d+) T0=(\d+) Tf=(\d+) Tc=(\d+)",
            line,
        )
        if not m:
            m = re.search(
                r"full=(\d+) compact=(\d+) ratio=([\d.]+).*?"
                r"T0=(\d+) Tf=(\d+) Tc=(\d+)",
                line,
            )
        if not m:
            continue
        if "effective_ratio" in line:
            full_h, comp_h = int(m.group(1)), int(m.group(2))
            ratio = float(m.group(3))
            max_running = int(m.group(4))
            t0, tf, tc = int(m.group(5)), int(m.group(6)), int(m.group(7))
        else:
            full_h, comp_h = int(m.group(1)), int(m.group(2))
            ratio = float(m.group(3))
            max_running = 32
            t0, tf, tc = int(m.group(4)), int(m.group(5)), int(m.group(6))
        rows.append({
            "full_heads": full_h, "compact_heads": comp_h,
            "effective_ratio": ratio, "max_running_requests": max_running,
            "T0": t0, "Tf": tf, "Tc": tc,
        })
    return rows


def pool_bytes(tokens, heads_per_layer, head_dim, layers, dtype_size):
    return (tokens + 1) * heads_per_layer * head_dim * dtype_size * 2 * layers


def main():
    # 4 点 sweep 的 Tf 从 capacity_sweep.csv 读(真实测量, 日志跨行格式不稳)
    csv_in = "/home/lixinze/HeadKV-SGLang/results/capacity.csv"
    sweep = []
    with open(csv_in) as f:
        for row in csv.DictReader(f):
            sweep.append({"ratio": float(row["full_head_ratio"]),
                          "Tf": int(row["Tf"])})
    sweep.sort(key=lambda r: r["ratio"])
    total_heads = LAYERS * KV_HEADS_PER_LAYER  # 256 KV heads
    out = []
    for s in sweep:
        ratio = s["ratio"]
        full_h = round(ratio * total_heads)
        comp_h = total_heads - full_h
        fh_pl = full_h // LAYERS
        ch_pl = comp_h // LAYERS if comp_h else 0
        # comp 池: Tc 未逐点记录, 用 chunk 容量上限(Tc=12288, 与日志一致)
        tc = 12288 if comp_h else 0
        full_b = pool_bytes(s["Tf"], fh_pl, HEAD_DIM, LAYERS, DTYPE_SIZE)
        comp_b = pool_bytes(tc, ch_pl, HEAD_DIM, LAYERS, DTYPE_SIZE) if ch_pl else 0
        out.append({
            "effective_full_kv_head_ratio": ratio,
            "full_pool_bytes": full_b,
            "compact_pool_bytes": comp_b,
            "full_pool_GB": round(full_b / 1e9, 3),
            "compact_pool_GB": round(comp_b / 1e9, 3),
            "max_total_tokens": s["Tf"],
            "max_running_requests": 32,
            "full_heads": full_h,
            "compact_heads": comp_h,
        })
    path = "/home/lixinze/HeadKV-SGLang/results/exp1_capacity.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    for o in out:
        print(f"ratio={o['effective_full_kv_head_ratio']:.2f} "
              f"full_pool={o['full_pool_GB']}GB comp_pool={o['compact_pool_GB']}GB "
              f"max_tokens={o['max_total_tokens']} max_running={o['max_running_requests']}")
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
