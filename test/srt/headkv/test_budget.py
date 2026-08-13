"""Budget 单元测试(纯算术)。"""
import pytest

from sglang.srt.headkv.budget import Budget, BudgetError, compute


class TestBudgetFormula:
    def test_budget_formula(self):
        # T0*(F+C) = Tf*F + Tc*C
        T0, F, C, R, V = 1000, 6, 2, 8, 32
        b = compute(T0, F, C, R, V)
        assert b.Tc == R * V == 256
        expected_tf = (T0 * (F + C) - b.Tc * C) // F
        assert b.Tf == expected_tf
        # 守恒等式
        assert b.Tf * F + b.Tc * C <= T0 * (F + C)
        assert b.Tf * F + b.Tc * C > T0 * (F + C) - F  # floor 误差 < 1 个 full slot

    def test_capacity_gain_when_compact_exists(self):
        # 存在 compact head → Tf > T0
        b = compute(100000, F=6 * 32, C=2 * 32, R=32, V=384)
        assert b.Tf > b.T0
        assert b.predicted_gain > 1.0

    def test_all_full_no_gain(self):
        # F = total → Tf == T0
        b = compute(100000, F=8 * 32, C=0, R=32, V=384)
        assert b.Tf == b.T0
        assert b.predicted_gain == 1.0

    def test_f0_rejected(self):
        with pytest.raises(BudgetError, match="F == 0"):
            compute(1000, F=0, C=8, R=8, V=32)

    def test_c0_degenerate(self):
        b = compute(1000, F=8, C=0, R=8, V=32)
        assert b.Tc == 0
        assert b.Tf == b.T0

    def test_impossible_config(self):
        # Tc*C >= T0*(F+C):R 巨大
        with pytest.raises(BudgetError, match="不可能"):
            compute(1000, F=8, C=8, R=100000, V=32)

    def test_invalid_inputs(self):
        with pytest.raises(BudgetError, match="T0"):
            compute(0, F=8, C=2, R=8, V=32)
        with pytest.raises(BudgetError, match="R"):
            compute(1000, F=8, C=2, R=0, V=32)
        with pytest.raises(BudgetError, match="V"):
            compute(1000, F=8, C=2, R=8, V=0)

    def test_realistic_headkv_scenario(self):
        # Llama-3.1-8B: 32 层 × 8 KV heads;full ratio 0.5 → F=C=128
        T0 = 204824
        F, C = 128, 128
        R, V = 32, 128 + 256
        b = compute(T0, F, C, R, V)
        assert b.Tc == 32 * 384 == 12288
        assert b.Tf > b.T0  # 容量收益
        print(f"  T0={b.T0} Tf={b.Tf} gain={b.predicted_gain:.3f}x")
