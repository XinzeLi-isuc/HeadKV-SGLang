"""RLKVPolicy:兼容原 RLKV adapter loader,去掉随机微扰。

- 输入:adapter_weights.tsv(fallback full_attention_heads.tsv)
- 二值化:sparsity-quantile(官方 RLKV 语义,threshold = quantile(scores, sparsity))
- 与原 `_load_rlkv_head_masks` 差异:
    1) 去掉 np.random.uniform(0, 1e-6) 随机微扰 → 确定性
    2) window 显式传入(不再读 fork 默认 16/32)
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch

from .config import HeadKVConfig, HeadKVConfigError
from .policy import HeadPolicy, model_num_kv_heads, model_num_layers
from .duo_policy import _stable_topk_mask


class RLKVPolicy(HeadPolicy):
    def __init__(self, cfg: HeadKVConfig, sparsity: float = 0.5):
        super().__init__(cfg)
        self.sparsity = sparsity
        self._mask: Optional[torch.Tensor] = None
        self._sink: Optional[int] = None
        self._recent: Optional[int] = None

    def load_global_kv_mask(self, model_config) -> torch.Tensor:
        if self._mask is not None:
            return self._mask
        path = self.cfg.pattern_path
        if not path or not os.path.isdir(path):
            raise HeadKVConfigError(f"adapter 目录不存在: {path!r}")
        tsv = os.path.join(path, "adapter_weights.tsv")
        if not os.path.isfile(tsv):
            tsv = os.path.join(path, "full_attention_heads.tsv")
        if not os.path.isfile(tsv):
            raise HeadKVConfigError(
                f"adapter 目录缺少 adapter_weights.tsv 或 full_attention_heads.tsv: {path!r}"
            )

        scores = np.loadtxt(tsv, dtype=float, delimiter="\t")
        scores = np.clip(scores, 0.0, 1.0)

        L = model_num_layers(model_config)
        G_kv = model_num_kv_heads(model_config, tp_size=1)
        if scores.shape != (L, G_kv):
            raise HeadKVConfigError(
                f"adapter shape {scores.shape} != 模型 ({L}, {G_kv})"
            )

        # 确定性二值化:sparsity-quantile(无随机微扰)
        s = self.sparsity
        if s >= 1:
            mask = np.zeros_like(scores, dtype=bool)
        elif s <= 0:
            mask = np.ones_like(scores, dtype=bool)
        else:
            thr = np.quantile(scores, s)
            mask = scores >= thr

        self._mask = torch.from_numpy(mask.astype(bool))
        self._sink = int(self.cfg.sink_size)
        self._recent = int(self.cfg.recent_size)
        return self._mask

    def sink_size(self) -> int:
        assert self._sink is not None, "先调用 load_global_kv_mask"
        return self._sink

    def recent_size(self) -> int:
        assert self._recent is not None, "先调用 load_global_kv_mask"
        return self._recent
