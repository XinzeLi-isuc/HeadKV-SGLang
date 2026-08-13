"""HeadKVConfig:HeadKV 参数模型与合法性校验。

window 优先级(禁止静默使用 RLKV fork 默认 16/32):
    CLI 显式 (sink_size / recent_size)
    > pattern config.json 的 deploy_sink_size / deploy_recent_size
    > pattern config.json 的 sink_size / recent_size
    > ValueError
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


class HeadKVConfigError(ValueError):
    """HeadKV 配置非法。"""


@dataclass
class HeadKVConfig:
    enable: bool = False
    policy: str = "duo"  # "duo" | "rlkv" | "manual"
    pattern_path: Optional[str] = None
    # 二值化:full_head_ratio 与 threshold 二选一;都未给 → pattern config.json 的 threshold
    full_head_ratio: Optional[float] = None
    threshold: Optional[float] = None
    # window 显式覆盖(优先级最高)
    sink_size: Optional[int] = None
    recent_size: Optional[int] = None
    # HeadKV 模式必填(拒绝 fork 默认 48 静默生效)
    max_running_requests: Optional[int] = None
    # 附加信息(来自 pattern config.json,由 policy 填充)
    config_sink_size: Optional[int] = None
    config_recent_size: Optional[int] = None
    config_threshold: Optional[float] = None

    def validate(self) -> None:
        """全部 fail-fast;返回 None 或抛 HeadKVConfigError。"""
        if not self.enable:
            return
        if self.policy not in ("duo", "rlkv", "manual"):
            raise HeadKVConfigError(f"policy 必须是 duo|rlkv|manual, got {self.policy!r}")
        if self.max_running_requests is None:
            raise HeadKVConfigError(
                "HeadKV 模式必须显式指定 --max-running-requests(拒绝 fork 默认 48)"
            )
        if self.max_running_requests <= 0:
            raise HeadKVConfigError(
                f"max_running_requests 必须 > 0, got {self.max_running_requests}"
            )
        if self.full_head_ratio is not None and self.threshold is not None:
            raise HeadKVConfigError(
                "full_head_ratio 与 threshold 互斥,只能指定其一"
            )
        if self.full_head_ratio is not None and not (0 < self.full_head_ratio <= 1):
            raise HeadKVConfigError(
                f"full_head_ratio 必须在 (0,1], got {self.full_head_ratio}"
            )
        # threshold 可为负(官方允许负阈值 = 全 full)
        if self.policy == "duo":
            if not self.pattern_path:
                raise HeadKVConfigError("duo policy 必须指定 --headkv-pattern-path")
        if self.policy == "rlkv":
            # rlkv policy 需要 adapter 目录(可走 server_args.adapter_load_path 兼容入口)
            if not self.pattern_path:
                raise HeadKVConfigError("rlkv policy 必须指定 adapter 目录")
        if self.sink_size is not None and self.sink_size < 0:
            raise HeadKVConfigError(f"sink_size 不能为负, got {self.sink_size}")
        if self.recent_size is not None and self.recent_size < 0:
            raise HeadKVConfigError(f"recent_size 不能为负, got {self.recent_size}")

    # ---- window 解析 ----
    def resolve_window(self) -> tuple[int, int]:
        """按优先级解析 sink/recent;无任何来源时抛错。"""
        sink = self.sink_size
        if sink is None:
            sink = self.config_sink_size
        if sink is None:
            raise HeadKVConfigError(
                "sink_size 未指定:显式 CLI > pattern deploy_sink_size > pattern sink_size;"
                "禁止回落到 RLKV fork 默认值"
            )
        recent = self.recent_size
        if recent is None:
            recent = self.config_recent_size
        if recent is None:
            raise HeadKVConfigError(
                "recent_size 未指定:显式 CLI > pattern deploy_recent_size > pattern "
                "recent_size;禁止回落到 RLKV fork 默认值"
            )
        return int(sink), int(recent)

    # ---- 二值化参数解析 ----
    def resolve_binarize(self) -> tuple[Optional[float], Optional[float]]:
        """返回 (full_head_ratio, threshold),至多一个非 None。"""
        if self.full_head_ratio is not None:
            return self.full_head_ratio, None
        if self.threshold is not None:
            return None, self.threshold
        # 都未给 → pattern config.json 的 threshold(官方默认 0.5)
        return None, self.config_threshold if self.config_threshold is not None else 0.5

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return (
            f"HeadKVConfig(enable={self.enable}, policy={self.policy!r}, "
            f"pattern_path={self.pattern_path!r}, full_head_ratio={self.full_head_ratio}, "
            f"threshold={self.threshold}, sink={self.sink_size}, recent={self.recent_size}, "
            f"max_running_requests={self.max_running_requests})"
        )
