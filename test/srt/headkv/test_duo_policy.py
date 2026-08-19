"""DuoAttentionPolicy 单元测试(纯 CPU,不依赖 GPU/SGLang)。"""
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from sglang.srt.headkv.config import HeadKVConfig, HeadKVConfigError
from sglang.srt.headkv.duo_policy import DuoAttentionPolicy, _stable_topk_mask

# Vendored copy of the official DuoAttention pattern for Meta-Llama-3.1-8B-Instruct
# (train config: lr=0.02, reg=0.05, ctx=1000..128000, multi-passkey10).
# Source: mit-han-lab/duo-attention release assets. See NOTICE for attribution.
DATA_DIR = Path(__file__).resolve().parent / "data" / "meta-llama-3.1-8b-instruct"
OFFICIAL_PATTERN = str(DATA_DIR)


class FakeModelConfig:
    num_hidden_layers = 32
    num_attention_heads = 32
    num_key_value_heads = 8

    def get_num_kv_heads(self, tp_size=1):
        return self.num_key_value_heads // tp_size


def _make_pattern_dir(scores: np.ndarray, sink=128, recent=256, threshold=0.5):
    d = tempfile.mkdtemp(prefix="headkv_pattern_")
    np.savetxt(os.path.join(d, "full_attention_heads.tsv"), scores, delimiter="\t")
    with open(os.path.join(d, "config.json"), "w") as f:
        f.write(
            f'{{"sink_size": {sink}, "recent_size": {recent}, "threshold": {threshold}}}'
        )
    return d


def _make_cfg(pattern_dir, **kw):
    base = dict(enable=True, policy="duo", pattern_path=pattern_dir,
                max_running_requests=32)
    base.update(kw)
    return HeadKVConfig(**base)


class TestLoadOfficialDuoPattern:
    def test_load_official_duo_pattern(self):
        cfg = _make_cfg(OFFICIAL_PATTERN)
        cfg.validate()
        pol = DuoAttentionPolicy(cfg)
        mask = pol.load_global_kv_mask(FakeModelConfig())
        assert mask.shape == (32, 8)
        assert mask.dtype == torch.bool
        assert pol.sink_size() == 128
        assert pol.recent_size() == 256

    def test_pattern_shape_validation(self):
        # 列数既非 kv(8) 也非 q(32)
        d = _make_pattern_dir(np.zeros((32, 5)))
        cfg = _make_cfg(d)
        pol = DuoAttentionPolicy(cfg)
        with pytest.raises(HeadKVConfigError, match="列数"):
            pol.load_global_kv_mask(FakeModelConfig())

    def test_row_mismatch(self):
        d = _make_pattern_dir(np.zeros((31, 8)))
        cfg = _make_cfg(d)
        pol = DuoAttentionPolicy(cfg)
        with pytest.raises(HeadKVConfigError, match="行数"):
            pol.load_global_kv_mask(FakeModelConfig())


class TestDeterministicTopk:
    def test_deterministic_topk(self):
        rng = np.random.default_rng(0)
        d = _make_pattern_dir(rng.random((32, 8)))
        cfg = _make_cfg(d, full_head_ratio=0.5)
        m1 = DuoAttentionPolicy(cfg).load_global_kv_mask(FakeModelConfig())
        m2 = DuoAttentionPolicy(cfg).load_global_kv_mask(FakeModelConfig())
        assert torch.equal(m1, m2)

    def test_topk_tie_break(self):
        # 所有分数相同 → 取前 k 个 head(head_id 小者优先,稳定)
        scores = np.full((2, 4), 0.5)
        mask = _stable_topk_mask(scores, 0.5)
        assert mask[0].tolist() == [True, True, False, False]
        assert mask[1].tolist() == [True, True, False, False]

    def test_topk_highest_scores_selected(self):
        scores = np.array([[0.1, 0.9, 0.2, 0.8]])
        mask = _stable_topk_mask(scores, 0.5)
        assert mask[0].tolist() == [False, True, False, True]

    def test_topk_ratio_one_all_full(self):
        scores = np.random.default_rng(1).random((3, 4))
        mask = _stable_topk_mask(scores, 1.0)
        assert mask.all()

    def test_full_head_ratio_vs_threshold_mutual_exclusion(self):
        d = _make_pattern_dir(np.zeros((32, 8)))
        cfg = _make_cfg(d, full_head_ratio=0.5, threshold=0.5)
        with pytest.raises(HeadKVConfigError, match="互斥"):
            cfg.validate()


class TestGqa:
    def test_gqa_kv_pattern_no_second_aggregation(self):
        # 8 列(KV 粒度)必须原样输出,禁止 OR 聚合
        rng = np.random.default_rng(2)
        scores = rng.random((32, 8))
        d = _make_pattern_dir(scores)
        cfg = _make_cfg(d, full_head_ratio=0.5)
        mask = DuoAttentionPolicy(cfg).load_global_kv_mask(FakeModelConfig())
        expected = _stable_topk_mask(scores, 0.5)
        assert torch.equal(mask, torch.from_numpy(expected))

    def test_qhead_pattern_or_aggregation(self):
        # 32 列(Q 粒度)→ OR 到 8 个 KV head
        rng = np.random.default_rng(3)
        scores = rng.random((32, 32))
        d = _make_pattern_dir(scores)
        cfg = _make_cfg(d, full_head_ratio=0.25)
        mask = DuoAttentionPolicy(cfg).load_global_kv_mask(FakeModelConfig())
        assert mask.shape == (32, 8)
        # 每个 KV group 有 4 个 Q head,ratio 0.25 → 每层 8 个 Q full → 期望 2 个 KV full
        per_layer = mask.sum(dim=1)
        assert (per_layer == 2).all()


class TestWindowPriority:
    def test_cli_overrides_config(self):
        d = _make_pattern_dir(np.zeros((32, 8)), sink=128, recent=256)
        cfg = _make_cfg(d, sink_size=16, recent_size=32)
        pol = DuoAttentionPolicy(cfg)
        pol.load_global_kv_mask(FakeModelConfig())
        assert (pol.sink_size(), pol.recent_size()) == (16, 32)

    def test_deploy_overrides_train(self):
        d = _make_pattern_dir(np.zeros((32, 8)), sink=128, recent=256)
        with open(os.path.join(d, "config.json"), "w") as f:
            f.write('{"deploy_sink_size": 64, "deploy_recent_size": 128, '
                    '"sink_size": 128, "recent_size": 256}')
        cfg = _make_cfg(d)
        pol = DuoAttentionPolicy(cfg)
        pol.load_global_kv_mask(FakeModelConfig())
        assert (pol.sink_size(), pol.recent_size()) == (64, 128)

    def test_config_json_fallback(self):
        d = _make_pattern_dir(np.zeros((32, 8)), sink=128, recent=256)
        cfg = _make_cfg(d)
        pol = DuoAttentionPolicy(cfg)
        pol.load_global_kv_mask(FakeModelConfig())
        assert (pol.sink_size(), pol.recent_size()) == (128, 256)

    def test_no_window_source_raises(self):
        d = _make_pattern_dir(np.zeros((32, 8)))
        # config.json 无 sink/recent 字段
        with open(os.path.join(d, "config.json"), "w") as f:
            f.write('{"threshold": 0.5}')
        cfg = _make_cfg(d)
        pol = DuoAttentionPolicy(cfg)
        with pytest.raises(HeadKVConfigError, match="sink_size 未指定"):
            pol.load_global_kv_mask(FakeModelConfig())


class TestThresholdMode:
    def test_negative_threshold_all_full(self):
        d = _make_pattern_dir(np.zeros((32, 8)), threshold=0.5)
        cfg = _make_cfg(d, threshold=-1.0)
        mask = DuoAttentionPolicy(cfg).load_global_kv_mask(FakeModelConfig())
        assert mask.all()

    def test_threshold_semantics(self):
        # 8 列(KV 粒度),threshold=0.5 → score >= 0.5 为 full
        scores8 = np.tile(np.array([0.1, 0.6, 0.5, 0.9, 0.2, 0.7, 0.4, 0.3]), (32, 1))
        d = _make_pattern_dir(scores8, threshold=0.5)
        cfg = _make_cfg(d, threshold=0.5)
        mask = DuoAttentionPolicy(cfg).load_global_kv_mask(FakeModelConfig())
        assert mask[0].tolist() == [False, True, True, True, False, True, False, False]  # >= 0.5


class TestConfigValidation:
    def test_max_running_requests_required(self):
        cfg = HeadKVConfig(enable=True, policy="duo",
                           pattern_path=OFFICIAL_PATTERN)
        with pytest.raises(HeadKVConfigError, match="max-running-requests"):
            cfg.validate()

    def test_pattern_path_required(self):
        cfg = HeadKVConfig(enable=True, policy="duo", max_running_requests=32)
        with pytest.raises(HeadKVConfigError, match="pattern-path"):
            cfg.validate()

    def test_ratio_range(self):
        d = _make_pattern_dir(np.zeros((32, 8)))
        cfg = _make_cfg(d, full_head_ratio=1.5)
        with pytest.raises(HeadKVConfigError, match="full_head_ratio"):
            cfg.validate()

    def test_missing_pattern_file(self):
        d = tempfile.mkdtemp()
        cfg = _make_cfg(d)
        pol = DuoAttentionPolicy(cfg)
        with pytest.raises(HeadKVConfigError, match="full_attention_heads.tsv"):
            pol.load_global_kv_mask(FakeModelConfig())
