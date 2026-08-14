"""S2:head 分布差异分析——Duo vs RLKV 的 full head 集合对比。

- 逐层 full 数分布
- 集合 Jaccard 重叠
- heatmap(层 × head)双图并排
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

FIG = "/home/lixinze/HeadKV-SGLang/figures"
os.makedirs(FIG, exist_ok=True)

DUO_PATTERN = "/home/lixinze/duo-attention-ref/attn_patterns/Meta-Llama-3.1-8B-Instruct/lr=0.02-reg=0.05-ctx=1000_128000-multi_passkey10/full_attention_heads.tsv"
RLKV_ADAPTER = "/home/lixinze/rlkv/head_dist/rlkv/Llama-3.1-8B-R1/llama_lr1e-2_ep2_bs32_reg1e-3_tau0.5/adapter_weights.tsv"


def binarize_duo(path, ratio=0.5):
    """DuoAttentionPolicy 语义:每层取 top-(ratio×G) 个 head。"""
    s = np.loadtxt(path, dtype=float, delimiter="\t")
    L, G = s.shape
    k = int(round(G * ratio))
    m = np.zeros((L, G), dtype=bool)
    for l in range(L):
        order = np.argsort(-s[l], kind="stable")[:k]
        m[l, order] = True
    return m


def binarize_rlkv(path, sparsity=0.5):
    s = np.loadtxt(path, dtype=float, delimiter="\t")
    s = np.clip(s, 0, 1)
    thr = np.quantile(s, sparsity)
    return s >= thr


def jaccard(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / union if union else 0.0


def main():
    duo = binarize_duo(DUO_PATTERN)
    rlkv = binarize_rlkv(RLKV_ADAPTER)
    L, G = duo.shape
    print(f"Duo : full={duo.sum()} ({duo.mean():.4f})")
    print(f"RLKV: full={rlkv.sum()} ({rlkv.mean():.4f})")
    print(f"Jaccard(full head 集合): {jaccard(duo, rlkv):.4f}")

    # 逐层 full 数
    per_layer = np.stack([duo.sum(axis=1), rlkv.sum(axis=1)], axis=1)
    print("\n逐层 full 数(前 8 层):")
    for l in range(min(8, L)):
        print(f"  layer {l:2d}: duo={per_layer[l,0]:2d} rlkv={per_layer[l,1]:2d}")

    # 每个 KV head 的层间一致性
    head_freq = np.stack([duo.sum(axis=0), rlkv.sum(axis=0)], axis=1)
    print("\n每个 KV head 的 full 层数(全部 8 head):")
    for h in range(G):
        print(f"  head {h}: duo={head_freq[h,0]:2d}/32 rlkv={head_freq[h,1]:2d}/32")

    # heatmap
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, m, title in ((axes[0], duo, "DuoAttention (ratio 0.5)"),
                         (axes[1], rlkv, "RLKV (sparsity 0.5)")):
        im = ax.imshow(m.astype(int), cmap="Reds", aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("KV head")
        ax.set_ylabel("Layer")
        ax.set_yticks(range(0, L, 4))
        fig.colorbar(im, ax=ax, fraction=0.02)
    fig.suptitle(f"Full-head distribution (Jaccard={jaccard(duo, rlkv):.3f})")
    fig.tight_layout()
    fig.savefig(f"{FIG}/s_head_distribution.png", dpi=150)
    print(f"\nsaved {FIG}/s_head_distribution.png")

    # 保存统计
    with open("/home/lixinze/HeadKV-SGLang/artifacts/s2_head_stats.json", "w") as f:
        json.dump({
            "duo_full": int(duo.sum()), "duo_ratio": round(float(duo.mean()), 4),
            "rlkv_full": int(rlkv.sum()), "rlkv_ratio": round(float(rlkv.mean()), 4),
            "jaccard": round(jaccard(duo, rlkv), 4),
            "per_layer_full": per_layer.tolist(),
            "head_freq": head_freq.tolist(),
        }, f, indent=2)
    print("saved artifacts/s2_head_stats.json")


if __name__ == "__main__":
    main()
