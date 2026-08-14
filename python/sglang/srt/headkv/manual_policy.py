"""ManualPolicy:人工指定 KV-head mask(调试/对照用)。

支持两种输入:
- mask_path:TSV/CSV 文件,内容为 0/1(bool)
- 全 full / 全 compact 快捷方式(full_head_ratio=1.0 / 0.0)
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch

from .config import HeadKVConfig, HeadKVConfigError
from .policy import HeadPolicy, model_num_kv_heads, model_num_layers


class ManualPolicy(HeadPolicy):
    def __init__(self, cfg: HeadKVConfig):
        super().__init__(cfg)
        self._mask: Optional[torch.Tensor] = None
        self._sink: Optional[int] = None
        self._recent: Optional[int] = None

    def load_global_kv_mask(self, model_config) -> torch.Tensor:
        if self._mask is not None:
            return self._mask
        L = model_num_layers(model_config)
        G_kv = model_num_kv_heads(model_config, tp_size=1)
        path = self.cfg.pattern_path
        if path and os.path.isfile(path):
            arr = np.loadtxt(path, delimiter=None)
            if arr.shape != (L, G_kv):
                raise HeadKVConfigError(
                    f"manual mask shape {arr.shape} != ({L}, {G_kv})"
                )
            mask = arr.astype(bool)
        elif self.cfg.full_head_ratio == 1.0:
            mask = np.ones((L, G_kv), dtype=bool)
        elif self.cfg.full_head_ratio == 0.0:
            mask = np.zeros((L, G_kv), dtype=bool)
        else:
            raise HeadKVConfigError(
                "manual policy 需要 mask 文件路径或 full_head_ratio ∈ {0.0, 1.0}"
            )
        self._mask = torch.from_numpy(mask)
        sink, recent = self.cfg.resolve_window()
        self._sink, self._recent = sink, recent
        return self._mask

    def sink_size(self) -> int:
        assert self._sink is not None, "先调用 load_global_kv_mask"
        return self._sink

    def recent_size(self) -> int:
        assert self._recent is not None, "先调用 load_global_kv_mask"
        return self._recent
