"""分析图表:单请求开销 vs 长度;并发时延 vs bs。"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG = "/home/lixinze/HeadKV-SGLang/figures"
A = "/home/lixinze/HeadKV-SGLang/artifacts"

# 1. 单请求 E2E vs 长度(绝对差恒定 0.33s)
exp_bc = json.load(open(f"{A}/exp_bc.json"))
lens = [r["len"] for r in exp_bc["exp_b"]]
f_t = [r["fullkv"]["median_s"] for r in exp_bc["exp_b"]]
d_t = [r["duokv"]["median_s"] for r in exp_bc["exp_b"]]

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(lens, f_t, "o-", label="FullKV (triton)", color="#2980b9", lw=2)
ax.plot(lens, d_t, "s-", label="DuoKV (ratio 0.5)", color="#c0392b", lw=2)
for x, f, d in zip(lens, f_t, d_t):
    ax.annotate(f"+{d-f:.2f}s", (x, (f + d) / 2), textcoords="offset points",
                xytext=(0, 12), ha="center", fontsize=9, color="#555")
ax.set_xlabel("Context length (tokens)")
ax.set_ylabel("E2E time (s, 3-run median, 32 decode tokens)")
ax.set_title("Single-request E2E: absolute overhead is constant ≈0.33s\n"
             "(fixed dual-dispatch cost, not per-token)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{FIG}/e2e_overhead.png", dpi=150)
print("saved", f"{FIG}/e2e_overhead.png")

# 2. 并发时延 vs bs
expc = json.load(open(f"{A}/exp_c_concurrency.json"))
bs = [r["bs"] for r in expc]
f_c = [r["fullkv"]["median_s"] for r in expc]
d_c = [r["duokv"]["median_s"] for r in expc]

fig2, ax2 = plt.subplots(figsize=(7, 4.5))
ax2.plot(bs, f_c, "o-", label="FullKV (triton)", color="#2980b9", lw=2)
ax2.plot(bs, d_c, "s-", label="DuoKV (ratio 0.5)", color="#c0392b", lw=2)
ax2.set_xlabel("Concurrent requests (8K ctx, 16 decode)")
ax2.set_ylabel("Median E2E (s)")
ax2.set_title("Concurrency sweep: prefill-wave bound, KV-pool agnostic\n"
              "(capacity edge lives in max_total_tokens & KV bytes)")
ax2.legend()
ax2.grid(alpha=0.3)
fig2.tight_layout()
fig2.savefig(f"{FIG}/concurrency_latency.png", dpi=150)
print("saved", f"{FIG}/concurrency_latency.png")
