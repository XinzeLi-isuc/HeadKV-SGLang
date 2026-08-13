"""Partition 单元测试(纯 CPU)。"""
import pytest
import torch

from sglang.srt.headkv.partition import to_tp_local


class TestPartition:
    def test_tp1_partition(self):
        mask = torch.tensor(
            [[1, 0, 1, 0, 1, 1, 0, 0],
             [0, 1, 0, 1, 0, 0, 1, 1]],
            dtype=torch.bool,
        )
        local = to_tp_local(mask, tp_rank=0, tp_size=1, num_kv_heads_per_tp=8)
        assert set(local.keys()) == {0, 1}
        assert torch.equal(local[0], mask[0].float())
        assert torch.equal(local[1], mask[1].float())

    def test_tp2_complete_and_disjoint(self):
        L, G = 4, 8
        rng = torch.Generator().manual_seed(0)
        mask = torch.randint(0, 2, (L, G), generator=rng, dtype=torch.bool)
        rank0 = to_tp_local(mask, 0, 2, 4)
        rank1 = to_tp_local(mask, 1, 2, 4)
        for l in range(L):
            # 完备:两 rank 拼接后 == global(切片本身保证不相交)
            union = torch.cat([rank0[l], rank1[l]])
            assert torch.equal(union, mask[l].float()), "并集 != global"
            assert rank0[l].shape == (4,) and rank1[l].shape == (4,)

    def test_shape_mismatch_raises(self):
        mask = torch.zeros((4, 8), dtype=torch.bool)
        with pytest.raises(ValueError, match="!= global kv heads"):
            to_tp_local(mask, 0, 2, 3)  # 2*3=6 != 8

    def test_tp_rank_out_of_range(self):
        mask = torch.zeros((4, 8), dtype=torch.bool)
        with pytest.raises(ValueError, match="越界"):
            to_tp_local(mask, 2, 2, 4)

    def test_1d_mask_rejected(self):
        with pytest.raises(ValueError, match="2D"):
            to_tp_local(torch.zeros(8), 0, 1, 8)
