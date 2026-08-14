"""S 级分析图表:CUDA Graph 开销消除 + 双 policy 质量对比。"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG = "/home/lixinze/HeadKV-SGLang/figures"

# 1. eager vs CG 绝对差
lens = [4096, 8192, 16384]
eager = [0.334, 0.335, 0.326]
cg = [0.030, 0.018, 0.047]

fig, ax = plt.subplots(figsize=(7, 4.5))
import numpy as np  # noqa: E402
x = np.arange(3)
w = 0.35
b1 = ax.bar(x - w / 2, eager, w, label="eager(固定启动成本)", color="#c0392b", alpha=0.85)
b2 = ax.bar(x + w / 2, cg, w, label="CUDA Graph", color="#2980b9", alpha=0.85)
for b in list(b1) + list(b2):
    ax.annotate(f"{b.get_height():.3f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                textcoords="offset points", xytext=(0, 3), ha="center", fontsize=9)
ax.set_xticks(x, [f"{l//1024}K" for l in lens])
ax.set_ylabel("Absolute E2E gap DuoKV − FullKV (s)")
ax.set_title("CUDA Graph eliminates 86~95% of the fixed dual-dispatch cost")
ax.legend()
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(f"{FIG}/s_cuda_graph_gap.png", dpi=150)
print("saved", f"{FIG}/s_cuda_graph_gap.png")

# 2. 双 policy 质量对比
lb = json.load(open("/home/lixinze/HeadKV-SGLang/artifacts/s1_quality.json"))
tasks = ["narrativeqa", "2wikimqa"]
f_duo = [lb[t]["f1_duo"] for t in tasks]
f_rlkv = [lb[t]["f1_rlkv"] for t in tasks]

fig2, ax2 = plt.subplots(figsize=(6.5, 4.2))
x2 = np.arange(len(tasks))
b1 = ax2.bar(x2 - w / 2, f_duo, w, label="DuoAttentionPolicy", color="#2980b9")
b2 = ax2.bar(x2 + w / 2, f_rlkv, w, label="RLKVPolicy", color="#e67e22")
for b in list(b1) + list(b2):
    ax2.annotate(f"{b.get_height():.4f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                 textcoords="offset points", xytext=(0, 3), ha="center", fontsize=9)
ax2.set_xticks(x2, tasks)
ax2.set_ylabel("F1 (30 samples, ≤16K ctx)")
ax2.set_title("LongBench: same effective ratio 0.5, Duo slightly ahead\n"
              "(paired t: 0.80 / 1.91 — not significant at n=30)")
ax2.legend()
ax2.grid(alpha=0.3, axis="y")
fig2.tight_layout()
fig2.savefig(f"{FIG}/s_policy_quality.png", dpi=150)
print("saved", f"{FIG}/s_policy_quality.png")
