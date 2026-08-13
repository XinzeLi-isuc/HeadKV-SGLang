"""HeadPolicy:算法无关的 head-wise KV 策略抽象。

Runtime 只消费 load_global_kv_mask() 的 bool mask(True=Full, False=Compact)
与 sink_size()/recent_size()。policy 差异只存在于这三个接口内;
runtime 中禁止出现 `if policy == "duo"` 之类分支。
"""
from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import torch

from .config import HeadKVConfig, HeadKVConfigError

if TYPE_CHECKING:  # pragma: no cover
    from .duo_policy import DuoAttentionPolicy
    from .rlkv_policy import RLKVPolicy


class HeadPolicy(abc.ABC):
    """head-wise KV 生命周期策略抽象。"""

    def __init__(self, cfg: HeadKVConfig):
        self.cfg = cfg

    @abc.abstractmethod
    def load_global_kv_mask(self, model_config) -> torch.Tensor:
        """返回 bool mask, shape=[num_layers, global_num_kv_heads]。
        True = Full(完整历史), False = Compact(sink + recent)。
        返回前必须通过维度校验(GQA 规则)。
        """

    @abc.abstractmethod
    def sink_size(self) -> int: ...

    @abc.abstractmethod
    def recent_size(self) -> int: ...

    @classmethod
    def create(cls, cfg: HeadKVConfig) -> "HeadPolicy":
        from .duo_policy import DuoAttentionPolicy
        from .manual_policy import ManualPolicy
        from .rlkv_policy import RLKVPolicy

        factories = {
            "duo": DuoAttentionPolicy,
            "rlkv": RLKVPolicy,
            "manual": ManualPolicy,
        }
        try:
            factory = factories[cfg.policy]
        except KeyError:
            raise HeadKVConfigError(f"未知 policy: {cfg.policy!r}") from None
        return factory(cfg)


# ---- 通用工具:模型配置鸭子类型取值 ----
def model_num_layers(model_config) -> int:
    return _getattr_chain(model_config, "num_hidden_layers", "num_layers")


def model_num_q_heads(model_config) -> int:
    """global Q head 数(TP=1 语义;多 TP 时调用方自行处理)。"""
    return _getattr_chain(model_config, "num_attention_heads", "num_q_heads")


def model_num_kv_heads(model_config, tp_size: int = 1) -> int:
    """global KV head 数。兼容 SGLang ModelConfig.get_num_kv_heads(tp) 与 HF config。"""
    fn = getattr(model_config, "get_num_kv_heads", None)
    if callable(fn):
        return int(fn(tp_size))
    return int(_getattr_chain(model_config, "num_key_value_heads", "num_kv_heads"))


def _getattr_chain(obj, *names):
    for n in names:
        v = getattr(obj, n, None)
        if v is not None:
            return v
    raise HeadKVConfigError(f"模型配置缺少属性: {names}")
