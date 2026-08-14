"""Budget:Full/Compact 双池 token capacity 计算。

保持 KV byte budget 近似不变(计划书 §3 公式):

    T0 × (F + C) = Tf × F + Tc × C
    Tc = R × V
    Tf = floor((T0 × (F + C) - Tc × C) / F)

边界(全部 fail-fast):
    F == 0          → 拒绝 serving 配置(MVP 直接报错)
    C == 0          → 退化为 FullKV(Tf = T0,不分配 comp pool)
    Tc*C >= T0*(F+C)→ 配置不可能,报错
    R 未显式        → HeadKVConfig.validate() 已拦截
"""
from __future__ import annotations

from dataclasses import dataclass, field


class BudgetError(ValueError):
    """预算配置非法。"""


@dataclass
class Budget:
    T0: int   # FullKV baseline token capacity
    F: int    # 全层 Full KV-head 数之和
    C: int    # 全层 Compact KV-head 数之和
    R: int    # max_running_requests
    V: int    # sink + recent
    Tc: int   # compact pool token slots = R × V
    Tf: int   # full pool token capacity
    predicted_gain: float = field(init=False)  # Tf / T0

    def __post_init__(self):
        self.predicted_gain = self.Tf / self.T0 if self.T0 > 0 else 0.0


def compute(T0: int, F: int, C: int, R: int, V: int) -> Budget:
    if T0 <= 0:
        raise BudgetError(f"T0 必须 > 0, got {T0}")
    if F < 0 or C < 0:
        raise BudgetError(f"F/C 不能为负: F={F}, C={C}")
    if R <= 0:
        raise BudgetError(f"R(max_running_requests) 必须 > 0, got {R}")
    if V <= 0:
        raise BudgetError(f"V(sink+recent) 必须 > 0, got {V}")
    if F == 0:
        raise BudgetError(
            "F == 0(无 full head):不作为正式 serving 配置,MVP 拒绝"
        )
    if C == 0:
        # 退化为 FullKV
        return Budget(T0=T0, F=F, C=0, R=R, V=V, Tc=0, Tf=T0)

    Tc = R * V
    if Tc * C >= T0 * (F + C):
        raise BudgetError(
            f"配置不可能: Tc*C({Tc*C}) >= T0*(F+C)({T0*(F+C)});"
            f"compact 池占用超出总预算。请减小 R 或 window"
        )
    Tf = (T0 * (F + C) - Tc * C) // F
    return Budget(T0=T0, F=F, C=C, R=R, V=V, Tc=Tc, Tf=Tf)
