"""DuoAttentionPolicy:加载官方 DuoAttention pattern 并确定性二值化。

- 输入目录须含 full_attention_heads.tsv + config.json
- 二值化两种模式(二选一):
    A) full_head_ratio R:稳定 top-k(每层取 R*G 个最高分 head;同分按 head_id 序)
    B) threshold T:      score >= T 为 Full(官方语义)
- GQA 维度校验:列数 == num_kv_heads 直接用(禁止二次 OR);
  列数 == num_q_heads 按共享 KV group OR;其他报错
- window 优先级见 config.HeadKVConfig.resolve_window()
"""
from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import torch

from .config import HeadKVConfig, HeadKVConfigError
from .policy import HeadPolicy, model_num_kv_heads, model_num_layers, model_num_q_heads

_PATTERN_FILENAME = "full_attention_heads.tsv"


class DuoAttentionPolicy(HeadPolicy):
    def __init__(self, cfg: HeadKVConfig):
        super().__init__(cfg)
        self._scores: Optional[np.ndarray] = None
        self._mask: Optional[torch.Tensor] = None
        self._sink: Optional[int] = None
        self._recent: Optional[int] = None

    # ---- HeadPolicy 接口 ----
    def load_global_kv_mask(self, model_config) -> torch.Tensor:
        if self._mask is not None:
            return self._mask
        self._load_pattern()
        self._validate_shape(model_config)
        self._binarize()
        self._resolve_window()
        return self._mask

    def sink_size(self) -> int:
        assert self._sink is not None, "先调用 load_global_kv_mask"
        return self._sink

    def recent_size(self) -> int:
        assert self._recent is not None, "先调用 load_global_kv_mask"
        return self._recent

    def summarize(self) -> dict:
        """返回可日志化的统计信息(Gate 2/3 启动日志用)。"""
        assert self._mask is not None and self._sink is not None, "先调用 load_global_kv_mask"
        total = self._mask.numel()
        full = int(self._mask.sum().item())
        return {
            "policy": "duo",
            "pattern_path": self.cfg.pattern_path,
            "mask_shape": list(self._mask.shape),
            "num_layers": int(self._mask.shape[0]),
            "num_kv_heads_per_layer": int(self._mask.shape[1]),
            "full_heads": full,
            "compact_heads": total - full,
            "nominal_full_ratio": self.cfg.full_head_ratio,
            "effective_full_ratio": round(full / total, 4),
            "binarize": ("topk" if self.cfg.full_head_ratio is not None else "threshold"),
            "threshold": self.cfg.threshold if self.cfg.threshold is not None
                         else self.cfg.config_threshold,
            "sink_size": self._sink,
            "recent_size": self._recent,
            "window_size": (self._sink or 0) + (self._recent or 0),
        }

    def _load_pattern(self) -> None:
        path = self.cfg.pattern_path
        if not path or not os.path.isdir(path):
            raise HeadKVConfigError(f"pattern 目录不存在: {path!r}")
        tsv = os.path.join(path, _PATTERN_FILENAME)
        if not os.path.isfile(tsv):
            raise HeadKVConfigError(
                f"pattern 目录缺少 {_PATTERN_FILENAME}: {path!r}"
            )
        cfg_json = os.path.join(path, "config.json")
        if not os.path.isfile(cfg_json):
            raise HeadKVConfigError(f"pattern 目录缺少 config.json: {path!r}")

        scores = np.loadtxt(tsv, dtype=float, delimiter="\t")
        if scores.ndim != 2:
            raise HeadKVConfigError(
                f"pattern 必须为二维 [layers, kv_heads], got shape={scores.shape}"
            )
        self._scores = np.clip(scores, 0.0, 1.0)

        with open(cfg_json) as f:
            j = json.load(f)
        self.cfg.config_sink_size = j.get("deploy_sink_size") or j.get("sink_size")
        self.cfg.config_recent_size = j.get("deploy_recent_size") or j.get("recent_size")
        self.cfg.config_threshold = j.get("threshold")

    def _validate_shape(self, model_config) -> None:
        L = model_num_layers(model_config)
        G_kv = model_num_kv_heads(model_config, tp_size=1)
        G_q = model_num_q_heads(model_config)
        rows, cols = self._scores.shape
        if rows != L:
            raise HeadKVConfigError(
                f"pattern 行数 {rows} != 模型层数 {L}"
            )
        if cols == G_kv:
            self._kv_mask = self._scores  # 已是 KV-head 粒度,禁止二次 OR
        elif cols == G_q:
            # Q-head 粒度 → 按共享 KV group OR 聚合
            assert G_q % G_kv == 0, f"GQA 不整分: q={G_q}, kv={G_kv}"
            groups = G_q // G_kv
            agg = self._scores.reshape(rows, G_kv, groups)
            self._kv_mask = agg.max(axis=2)  # OR: 任一 Q head 为 full → KV head full
        else:
            raise HeadKVConfigError(
                f"pattern 列数 {cols} 既不等于 num_kv_heads({G_kv}) "
                f"也不等于 num_q_heads({G_q}),禁止猜测映射"
            )

    def _binarize(self) -> None:
        ratio, thr = self.cfg.resolve_binarize()
        if ratio is not None:
            mask = _stable_topk_mask(self._kv_mask, ratio)
        else:
            mask = self._kv_mask >= thr
        self._mask = torch.from_numpy(mask.astype(bool))

    def _resolve_window(self) -> None:
        sink, recent = self.cfg.resolve_window()
        self._sink, self._recent = sink, recent


def _stable_topk_mask(scores: np.ndarray, ratio: float) -> np.ndarray:
    """确定性 top-k:每层保留最高分的前 round(ratio*G) 个 head。

    同分打破:head_id 小的优先(np.lexsort 稳定)。
    返回 bool [L, G]。
    """
    L, G = scores.shape
    mask = np.zeros_like(scores, dtype=bool)
    head_ids = np.arange(G)
    for l in range(L):
        k = int(round(ratio * G))
        if k <= 0:
            continue
        if k >= G:
            mask[l] = True
            continue
        # 主键 -score(升序 = score 降序),次键 head_id;取前 k 个
        order = np.lexsort((head_ids, -scores[l]))
        mask[l, order[:k]] = True
    return mask
