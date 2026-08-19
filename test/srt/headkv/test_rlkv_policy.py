"""RLKVPolicy 单元测试(纯 CPU)。

覆盖:adapter 加载、sparsity-quantile 二值化语义、确定性(无随机微扰)、
shape 校验、与 DuoAttentionPolicy 的 mask 差异、window/sparsity 参数传递。
"""
import numpy as np
import pytest
import torch

from sglang.srt.headkv.config import HeadKVConfig, HeadKVConfigError
from sglang.srt.headkv.duo_policy import DuoAttentionPolicy
from sglang.srt.headkv.policy import HeadPolicy
from sglang.srt.headkv.rlkv_policy import RLKVPolicy

L, G = 32, 8


class FakeModelConfig:
    num_hidden_layers = L
    num_attention_heads = 32
    num_key_value_heads = G
    def get_num_kv_heads(self, tp_size=1):
        return G // tp_size


def _make_cfg(adapter_dir, sparsity=None, sink=16, recent=32):
    return HeadKVConfig(
        enable=True, policy="rlkv", pattern_path=adapter_dir,
        sparsity=sparsity, sink_size=sink, recent_size=recent,
        max_running_requests=16,
    )


def _make_adapter(tmp_path, scores=None, fname="adapter_weights.tsv"):
    if scores is None:
        rng = np.random.default_rng(0)
        scores = rng.uniform(0.1, 0.9, (L, G))
    p = tmp_path / fname
    np.savetxt(p, scores, delimiter="\t")
    return str(tmp_path), scores


class TestRLKVPolicyLoad:
    def test_load_adapter_and_shape(self, tmp_path):
        d, _ = _make_adapter(tmp_path)
        p = RLKVPolicy(_make_cfg(d))
        mask = p.load_global_kv_mask(FakeModelConfig())
        assert mask.shape == (L, G)
        assert mask.dtype == torch.bool

    def test_fallback_full_attention_heads(self, tmp_path):
        d, _ = _make_adapter(tmp_path, fname="full_attention_heads.tsv")
        p = RLKVPolicy(_make_cfg(d))
        mask = p.load_global_kv_mask(FakeModelConfig())
        assert mask.shape == (L, G)

    def test_missing_dir_rejected(self, tmp_path):
        p = RLKVPolicy(_make_cfg(str(tmp_path / "nope")))
        with pytest.raises(HeadKVConfigError, match="adapter"):
            p.load_global_kv_mask(FakeModelConfig())

    def test_shape_mismatch_rejected(self, tmp_path):
        d, _ = _make_adapter(tmp_path, scores=np.ones((16, 4)))
        p = RLKVPolicy(_make_cfg(d))
        with pytest.raises(HeadKVConfigError, match="shape"):
            p.load_global_kv_mask(FakeModelConfig())


class TestRLKVPolicyBinarization:
    def test_sparsity_quantile_semantics(self, tmp_path):
        """threshold = quantile(scores, sparsity);mask = scores >= threshold。"""
        scores = np.tile(np.arange(8, dtype=float) / 8, (L, 1))  # 0.0..0.875
        d, _ = _make_adapter(tmp_path, scores=scores)
        p = RLKVPolicy(_make_cfg(d, sparsity=0.5))
        mask = p.load_global_kv_mask(FakeModelConfig())
        # quantile(0.5) = 0.4375 → 0.5..0.875 为 full = 4/8 每层
        assert mask.float().mean().item() == pytest.approx(0.5, abs=1e-6)
        assert mask[0, 4:].all() and not mask[0, :4].any()

    def test_sparsity_1_all_compact(self, tmp_path):
        d, _ = _make_adapter(tmp_path)
        p = RLKVPolicy(_make_cfg(d, sparsity=1.0))
        mask = p.load_global_kv_mask(FakeModelConfig())
        assert not mask.any()

    def test_sparsity_0_all_full(self, tmp_path):
        d, _ = _make_adapter(tmp_path)
        p = RLKVPolicy(_make_cfg(d, sparsity=0.0))
        mask = p.load_global_kv_mask(FakeModelConfig())
        assert mask.all()

    def test_sparsity_range_validated(self, tmp_path):
        d, _ = _make_adapter(tmp_path)
        cfg = _make_cfg(d, sparsity=1.5)
        with pytest.raises(HeadKVConfigError, match="sparsity"):
            cfg.validate()

    def test_default_sparsity_0_5(self, tmp_path):
        d, scores = _make_adapter(tmp_path)
        p = RLKVPolicy(_make_cfg(d))  # sparsity=None → 0.5
        assert p.sparsity == 0.5


class TestRLKVPolicyDeterminism:
    def test_deterministic_no_random_perturbation(self, tmp_path):
        """两次加载 mask 完全一致(原 loader 有 np.random.uniform 微扰)。"""
        d, _ = _make_adapter(tmp_path)
        p1 = RLKVPolicy(_make_cfg(d, sparsity=0.5))
        p2 = RLKVPolicy(_make_cfg(d, sparsity=0.5))
        m1 = p1.load_global_kv_mask(FakeModelConfig())
        m2 = p2.load_global_kv_mask(FakeModelConfig())
        assert torch.equal(m1, m2)

    def test_factory_dispatch_rlkv(self, tmp_path):
        d, _ = _make_adapter(tmp_path)
        cfg = _make_cfg(d)
        policy = HeadPolicy.create(cfg)
        assert isinstance(policy, RLKVPolicy)


class TestRLKVvsDuo:
    def test_masks_differ(self, tmp_path):
        """RLKV(quantile)与 Duo(topk)对同一 score 矩阵产生不同 mask(边界同分除外)。"""
        rng = np.random.default_rng(1)
        scores = rng.uniform(0.3, 0.7, (L, G))
        d, _ = _make_adapter(tmp_path, scores=scores)
        import json
        # duo 用同名 scores 的 full_attention_heads.tsv + config.json
        np.savetxt(f"{tmp_path}/full_attention_heads.tsv", scores, delimiter="\t")
        with open(f"{tmp_path}/config.json", "w") as f:
            json.dump({"sink_size": 16, "recent_size": 32}, f)
        r = RLKVPolicy(_make_cfg(d, sparsity=0.5))
        du = DuoAttentionPolicy(HeadKVConfig(
            enable=True, policy="duo", pattern_path=d, threshold=0.5,
            sink_size=16, recent_size=32, max_running_requests=16,
        ))
        m_r = r.load_global_kv_mask(FakeModelConfig())
        m_d = du.load_global_kv_mask(FakeModelConfig())
        assert not torch.equal(m_r, m_d)  # 两种算法 head 选择不同
        # 但都保留约 50%(threshold 0.5 在均匀分数上 ≈ quantile 0.5)
        assert m_r.float().mean().item() == pytest.approx(0.5, abs=0.05)
        assert m_d.float().mean().item() == pytest.approx(0.5, abs=0.05)


class TestRLKVWindow:
    def test_window_from_cfg(self, tmp_path):
        d, _ = _make_adapter(tmp_path)
        p = RLKVPolicy(_make_cfg(d, sink=16, recent=32))
        p.load_global_kv_mask(FakeModelConfig())
        assert p.sink_size() == 16
        assert p.recent_size() == 32

    def test_summarize(self, tmp_path):
        d, _ = _make_adapter(tmp_path)
        p = RLKVPolicy(_make_cfg(d, sparsity=0.5))
        p.load_global_kv_mask(FakeModelConfig())
        s = p.summarize()
        assert s["policy"] == "rlkv"
        assert s["mask_shape"] == [L, G]
        assert s["nominal_full_ratio"] == 0.5
        assert abs(s["effective_full_ratio"] - 0.5) < 0.05
        assert s["window_size"] == 48
