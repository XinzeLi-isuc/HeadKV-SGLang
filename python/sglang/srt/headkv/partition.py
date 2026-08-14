"""Partition:global KV-head mask → TP-local mask。

MVP 仅 TP=1(恒等映射),接口保持通用。划分语义与 fork
`_load_rlkv_head_masks`(model_runner.py L1783-1797)一致:
start = tp_rank * num_kv_heads_per_tp,取连续切片。

不变量:各 rank mask 并集 == global mask,交集为空(完备且不相交)。
"""
from __future__ import annotations

from typing import Dict

import torch


def to_tp_local(
    global_mask: torch.Tensor,
    tp_rank: int,
    tp_size: int,
    num_kv_heads_per_tp: int,
) -> Dict[int, torch.Tensor]:
    """global_mask: [L, global_kv_heads] bool(或 float 0/1)。

    返回 {layer_id: float32 tensor [num_kv_heads_per_tp]},与 fork 现有
    head_masks 格式一致(1.0 = full, 0.0 = compact)。
    """
    if global_mask.ndim != 2:
        raise ValueError(f"global_mask 必须为 2D [L, G_kv], got {global_mask.shape}")
    L, G_kv = global_mask.shape
    if tp_size * num_kv_heads_per_tp != G_kv:
        raise ValueError(
            f"tp_size({tp_size}) × num_kv_heads_per_tp({num_kv_heads_per_tp}) "
            f"!= global kv heads({G_kv})"
        )
    if not (0 <= tp_rank < tp_size):
        raise ValueError(f"tp_rank {tp_rank} 越界 [0, {tp_size})")

    start = tp_rank * num_kv_heads_per_tp
    end = start + num_kv_heads_per_tp
    slice_ = global_mask[:, start:end]

    local = {}
    for l in range(L):
        local[l] = slice_[l].to(torch.float32)
    return local
