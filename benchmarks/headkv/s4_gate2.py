"""S4 Gate 2/4:current main 双池初始化验证(物理 shape + byte accounting)。

需 PYTHONPATH=/home/lixinze/sglang-main/python(覆盖 rlkv-eval 的 v0.5.2 editable)。
"""
import sys

import torch

sys.path.insert(0, "/home/lixinze/sglang-main/python")

from sglang.srt.headkv.partition import to_tp_local  # noqa: E402
from sglang.srt.mem_cache.headkv_pool import (  # noqa: E402
    HeadReallocAllocator,
    HeadReallocKVPool,
)

DUO = "/home/lixinze/duo-attention-ref/attn_patterns/Meta-Llama-3.1-8B-Instruct/lr=0.02-reg=0.05-ctx=1000_128000-multi_passkey10"

DEV = "cuda:0"
L, G, HEAD_DIM = 32, 8, 128
SIZE_FULL, SIZE_COMP = 4096, 32 * 384  # R=32, V=384


def load_mask():
    import numpy as np

    s = np.loadtxt(f"{DUO}/full_attention_heads.tsv", dtype=float, delimiter="\t")
    assert s.shape == (L, G)
    k = 4  # ratio 0.5 → 每层 4 full
    mask = torch.zeros((L, G), dtype=torch.bool)
    for l in range(L):
        order = np.argsort(-s[l], kind="stable")[:k]
        mask[l, order] = True
    return to_tp_local(mask, 0, 1, G)


def main():
    assert torch.cuda.is_available(), "需要 GPU"
    masks = load_mask()
    print(f"[Gate2] mask: 32 层, 每层 full={int(masks[0].sum())}/8")

    pool = HeadReallocKVPool(
        size_full=SIZE_FULL, size_comp=SIZE_COMP, head_masks=masks,
        head_dim=HEAD_DIM, layer_num=L, dtype=torch.bfloat16, device=DEV,
        enable_memory_saver=False,
    )
    print(f"[Gate2] pool 构造 OK: size_full={pool.size_full} "
          f"size_comp={pool.size_comp}")

    # 物理 shape:full[Tf+1, n_full, dim] / comp[Tc+1, n_comp, dim]
    fb = pool.full_k_buffer[0]
    cb = pool.comp_k_buffer[0]
    print(f"[Gate4] full_k[0] shape={tuple(fb.shape)} (期望 {(SIZE_FULL+1, 4, HEAD_DIM)})")
    print(f"[Gate4] comp_k[0] shape={tuple(cb.shape)} (期望 {(SIZE_COMP+1, 4, HEAD_DIM)})")
    assert fb.shape == (SIZE_FULL + 1, 4, HEAD_DIM), fb.shape
    assert cb.shape == (SIZE_COMP + 1, 4, HEAD_DIM), cb.shape
    assert pool.get_value_buffer(0).shape == (SIZE_FULL + 1, 4, HEAD_DIM)

    # byte accounting
    k_bytes, v_bytes = pool.get_kv_size_bytes()
    print(f"[Gate4] bytes: K={k_bytes/1e9:.2f}GB V={v_bytes/1e9:.2f}GB")
    per_head = HEAD_DIM * 2  # bf16
    expect_k = (SIZE_FULL + 1) * 4 * per_head * L + (SIZE_COMP + 1) * 4 * per_head * L
    assert k_bytes == expect_k, (k_bytes, expect_k)
    print("[Gate4] byte accounting 逐位吻合")

    # allocator:comp chunk 计数 + alloc/free 循环
    allocator = HeadReallocAllocator(
        SIZE_FULL, SIZE_COMP, torch.bfloat16, DEV, pool, need_sort=False,
        window_size=384,
    )
    print(f"[Gate2] allocator OK: max_comp_chunks={allocator.max_comp_chunks} "
          f"(期望 {SIZE_COMP//384})")
    assert allocator.max_comp_chunks == SIZE_COMP // 384 == 32

    # alloc/free 循环恢复
    locs = allocator.alloc(100)
    assert len(locs) == 100
    cb0 = allocator.alloc_comp_window()
    assert cb0 == 1 + 31 * 384  # LIFO pop → 最后一个 chunk
    allocator.free_comp_window(cb0)
    allocator.free(locs)
    assert allocator.available_size() == SIZE_FULL
    assert allocator.comp_chunks_available() == allocator.max_comp_chunks
    print("[Gate2] alloc/free 循环恢复初始态 OK")

    print("\n[Gate2/4] PASS: current main 双池初始化 + 物理 shape 验证通过")


if __name__ == "__main__":
    main()
