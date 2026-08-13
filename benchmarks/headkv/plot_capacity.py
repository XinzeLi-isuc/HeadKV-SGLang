"""Phase 7 图表:Max KV Token Capacity vs Effective Full-head Ratio。"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG_DIR = "/home/lixinze/HeadKV-SGLang/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# capacity sweep(实测,artifacts/capacity_sweep.csv + ratio=1.0 = T0)
ratios = [0.25, 0.50, 0.75, 1.00]
tf = [782432, 397360, 269002, 204824]
t0 = 204824

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(ratios, [x / 1e3 for x in tf], "o-", color="#c0392b", lw=2, ms=7,
        label="DuoKV (实测)")
ax.axhline(t0 / 1e3, color="#7f8c8d", ls="--", lw=1.5,
           label=f"FullKV baseline (T0={t0})")
for x, y in zip(ratios, tf):
    ax.annotate(f"{y}", (x, y / 1e3), textcoords="offset points",
                xytext=(0, 10), ha="center", fontsize=9)
ax.set_xlabel("Effective full-head ratio")
ax.set_ylabel("Max KV token capacity (×1000)")
ax.set_title("Max KV Token Capacity vs Full-head Ratio\n"
             "(Meta-Llama-3.1-8B-Instruct, A6000, mem-fraction=0.85)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/capacity_vs_ratio.png", dpi=150)
print(f"saved {FIG_DIR}/capacity_vs_ratio.png")

# 增益 vs ratio
fig2, ax2 = plt.subplots(figsize=(7, 4.5))
gains = [x / t0 for x in tf]
ax2.bar([str(r) for r in ratios], gains, color="#2980b9", alpha=0.85)
for i, g in enumerate(gains):
    ax2.annotate(f"{g:.2f}x", (i, g), textcoords="offset points",
                 xytext=(0, 6), ha="center", fontsize=10)
ax2.axhline(1.0, color="#7f8c8d", ls="--", lw=1)
ax2.set_xlabel("Effective full-head ratio")
ax2.set_ylabel("Capacity gain (Tf / T0)")
ax2.set_title("KV Token Capacity Gain vs Full-head Ratio")
ax2.grid(alpha=0.3, axis="y")
fig2.tight_layout()
fig2.savefig(f"{FIG_DIR}/capacity_gain.png", dpi=150)
print(f"saved {FIG_DIR}/capacity_gain.png")
