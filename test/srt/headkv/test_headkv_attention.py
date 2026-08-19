"""HeadKV attention 语义测试(纯 CPU,current SGLang main 移植版)。

验证 full_to_comp_mapping 的 sink/recent/环形语义与 ref_attention
窗口数学一致。forward 级语义由 E2E(server)验证。

基线:rlkv-sglang-v0.5.2 test/srt/headkv/test_headkv_attention.py,
仅适配 current-main 的模块路径与 backend 接口。
"""
import numpy as np
import pytest
import torch

from ref_attention import ref_single_layer_attention  # noqa: E402

from sglang.srt.headkv.partition import to_tp_local  # noqa: E402
from sglang.srt.layers.attention.headkv_backend import (  # noqa: E402
    HeadReallocAttnBackend,
)
from sglang.srt.mem_cache.headkv_pool import (  # noqa: E402
    HeadReallocAllocator,
    HeadReallocKVPool,
)

SINK, RECENT = 8, 16
V = SINK + RECENT
N_LAYERS = 1
Q_HEADS, KV_HEADS, HEAD_DIM = 4, 2, 32


class FakeArgs:
    sink_window_size = SINK
    recent_window_size = RECENT
    triton_attention_num_kv_splits = 8


class FakeModelConfig:
    num_attention_heads = Q_HEADS
    num_key_value_heads = KV_HEADS
    context_len = 512
    head_dim = HEAD_DIM

    def get_num_kv_heads(self, tp_size: int = 1):
        return KV_HEADS // tp_size


class FakeReqToTokenPool:
    def __init__(self, size):
        self.size = size
        self.req_to_token = torch.zeros(size, 512, dtype=torch.int64)


class FakeRunner:
    def __init__(self, mask):
        self.device = "cpu"
        self.gpu_id = 0
        self.server_args = FakeArgs()
        self.model_config = FakeModelConfig()
        self.req_to_token_pool = FakeReqToTokenPool(4)
        self.tp_rank = 0
        self.headkv_sink_size = SINK
        self.headkv_recent_size = RECENT
        self.token_to_kv_pool = None
        self.token_to_kv_pool_allocator = None


def _make_env(mask_layer):
    """构造 CPU 双池环境,返回 (runner, pool, allocator, backend)。"""
    masks = to_tp_local(
        torch.tensor([mask_layer], dtype=torch.bool), 0, 1, KV_HEADS
    )
    runner = FakeRunner(masks)
    pool = HeadReallocKVPool(
        size_full=1024, size_comp=4 * V, head_masks=masks,
        head_dim=HEAD_DIM, layer_num=N_LAYERS, dtype=torch.float32,
        device="cpu", enable_memory_saver=False,
    )
    runner.token_to_kv_pool = pool
    allocator = HeadReallocAllocator(
        1024, 4 * V, torch.float32, "cpu", pool, need_sort=False,
        window_size=V,
    )
    runner.token_to_kv_pool_allocator = allocator
    backend = HeadReallocAttnBackend(runner, masks)
    backend._allocator_ref = allocator
    return runner, pool, allocator, backend


def _extend(batch, pool, allocator, backend, req_idx, prefix_len, seq_len):
    """模拟一次 extend:分配 full locs + backend 自动分配 comp chunk 并更新 mapping。

    返回 (full_locs, comp_base)。comp_base 从 position-0 mapping 反推。
    """
    req_to_token = backend.req_to_token
    need = seq_len - prefix_len
    full_locs = allocator.alloc(need)
    req_to_token[req_idx, prefix_len:seq_len] = torch.tensor(full_locs)

    class FB:
        batch_size = 1
        req_pool_indices = torch.tensor([req_idx])
        extend_prefix_lens = torch.tensor([prefix_len])
        seq_lens = torch.tensor([seq_len])

    fb = FB()
    backend._update_comp_mapping_extend(pool, fb)

    first_full_loc = req_to_token[req_idx, 0].item()
    first_map = pool.full_to_comp_mapping[first_full_loc].item()
    assert first_map > 0, "首次 extend 应自动分配 comp chunk"
    comp_base = (first_map - 1) // V * V + 1
    return full_locs, comp_base


def _decode(batch, pool, allocator, backend, req_idx, seq_len):
    """模拟一次 decode:新 token 位置 seq_len(seq_lens 已递增为 seq_len+1 的调用约定)。"""
    class FB:
        batch_size = 1
        req_pool_indices = torch.tensor([req_idx])
        seq_lens = torch.tensor([seq_len])

    fb = FB()
    backend._update_comp_mapping_decode(pool, fb.req_pool_indices, fb.seq_lens, 1)


class TestExtendMapping:
    def test_extend_long_seq_sink_recent_only(self):
        _, pool, allocator, backend = _make_env([1, 0])
        full_locs, comp_base = _extend(None, pool, allocator, backend, 0, 0, 60)
        mapping = pool.full_to_comp_mapping[full_locs]
        # sink 0..7 映射;recent 44..59 映射;中间 8..43 不映射
        assert (mapping[:SINK] > 0).all()
        assert (mapping[SINK:44] == 0).all()
        assert (mapping[44:] > 0).all()
        # 槽位公式:comp_base + pos(前 8);comp_base + sink + (pos - sink) % recent
        assert mapping[3].item() == comp_base + 3
        assert mapping[50].item() == comp_base + SINK + (50 - SINK) % RECENT

    def test_extend_short_seq_all_mapped(self):
        _, pool, allocator, backend = _make_env([1, 0])
        full_locs, comp_base = _extend(None, pool, allocator, backend, 0, 0, 10)
        mapping = pool.full_to_comp_mapping[full_locs]
        assert (mapping > 0).all()  # L <= V:全部映射
        assert mapping[9].item() == comp_base + 9

    def test_comp_base_reused_across_chunks(self):
        _, pool, allocator, backend = _make_env([1, 0])
        _, cb1 = _extend(None, pool, allocator, backend, 0, 0, 20)
        _, cb2 = _extend(None, pool, allocator, backend, 0, 20, 40)
        assert cb1 == cb2  # 同一 request 复用同一 chunk

    def test_chunked_extend_recent_moves(self):
        """chunked extend 后,最终 recent 窗口是相对最终 seq_len 的。"""
        _, pool, allocator, backend = _make_env([1, 0])
        full_locs_2, _ = _extend(None, pool, allocator, backend, 0, 0, 30)
        full_locs_3, _ = _extend(None, pool, allocator, backend, 0, 30, 60)
        mapping = pool.full_to_comp_mapping[full_locs_3]
        # 第二次 extend 的 token 30..59:44..59 应为 recent(相对 60)
        assert (mapping[14:] > 0).all()
        assert (mapping[:14] == 0).all()

    def test_extend_window_math_matches_ref(self):
        """mapping 槽位集合 == ref_attention 的窗口 mask 列集合。"""
        _, pool, allocator, backend = _make_env([1, 0])
        full_locs, comp_base = _extend(None, pool, allocator, backend, 0, 0, 60)
        mapping = pool.full_to_comp_mapping[full_locs]
        mapped = set((mapping > 0).nonzero().flatten().tolist())
        expected = set(range(SINK)) | set(range(60 - RECENT, 60))
        assert mapped == expected


class TestDecodeMapping:
    def test_decode_ring_wraparound(self):
        _, pool, allocator, backend = _make_env([1, 0])
        full_locs, comp_base = _extend(None, pool, allocator, backend, 0, 0, 30)
        req_to_token = backend.req_to_token
        for new_len in range(31, 41):
            pos = new_len - 1
            full_loc = allocator.alloc(1)[0]
            req_to_token[0, pos] = full_loc
            _decode(None, pool, allocator, backend, 0, new_len)
            got = pool.full_to_comp_mapping[full_loc].item()
            expected = comp_base + SINK + (pos - SINK) % RECENT
            assert got == expected, f"pos={pos}"

    def test_decode_before_sink_uses_sink_slots(self):
        _, pool, allocator, backend = _make_env([1, 0])
        _, comp_base = _extend(None, pool, allocator, backend, 0, 0, 5)
        req_to_token = backend.req_to_token
        for new_len in range(6, 9):  # pos 5..7 < sink=8
            pos = new_len - 1
            full_loc = allocator.alloc(1)[0]
            req_to_token[0, pos] = full_loc
            _decode(None, pool, allocator, backend, 0, new_len)
            assert pool.full_to_comp_mapping[full_loc].item() == comp_base + pos

    def test_ring_overwrites_old_recent(self):
        """recent 环:相隔 recent 的 token 映射同一 comp 槽(后者覆盖前者)。"""
        _, pool, allocator, backend = _make_env([1, 0])
        _, comp_base = _extend(None, pool, allocator, backend, 0, 0, 30)
        req_to_token = backend.req_to_token
        loc_a = allocator.alloc(1)[0]
        req_to_token[0, 30] = loc_a
        _decode(None, pool, allocator, backend, 0, 31)
        slot_a = pool.full_to_comp_mapping[loc_a].item()
        loc_b = allocator.alloc(1)[0]
        req_to_token[0, 46] = loc_b
        _decode(None, pool, allocator, backend, 0, 47)
        slot_b = pool.full_to_comp_mapping[loc_b].item()
        assert slot_a == comp_base + SINK + (30 - SINK) % RECENT
        assert slot_b == comp_base + SINK + (46 - SINK) % RECENT
        assert slot_a == slot_b  # 环形覆盖:同一槽


class TestRefAttention:
    def test_ref_full_head_causal(self):
        q = torch.randn(1, 4, 32)
        k = torch.randn(8, 4, 32)
        v = torch.randn(8, 4, 32)
        mask = torch.ones(4, dtype=torch.bool)
        out = ref_single_layer_attention(q, k, v, mask, 2, 4, 1)
        assert out.shape == (1, 4, 32)

    def test_ref_comp_head_window_only(self):
        """comp head:只有 sink+recent 的 KV 有贡献,中间 token 完全不可见。"""
        L, sink, recent = 20, 4, 6
        q = torch.randn(1, 1, 16)
        k = torch.randn(L, 1, 16)
        v = torch.zeros(L, 1, 16)
        v[:sink] = 1.0
        v[L - recent:] = 2.0
        v[sink:L - recent] = 5.0
        out = ref_single_layer_attention(
            q, k, v, torch.tensor([False]), sink, recent, 1
        )
        assert out.min().item() >= 1.0 - 1e-5
        assert out.max().item() <= 2.0 + 1e-5

    def test_ref_gqa_expansion(self):
        L, sink, recent = 12, 3, 4
        q = torch.randn(1, 4, 16)  # 4 Q heads
        k = torch.randn(L, 2, 16)  # 2 KV heads
        v = torch.randn(L, 2, 16)
        mask = torch.tensor([True, False, False, True])
        out = ref_single_layer_attention(q, k, v, mask, sink, recent, 2)
        assert out.shape == (1, 4, 16)

    def test_ref_matches_numpy_manual(self):
        torch.manual_seed(0)
        L = 6
        q = torch.randn(1, 1, 8)
        k = torch.randn(L, 1, 8)
        v = torch.randn(L, 1, 8)
        mask = torch.ones(1, dtype=torch.bool)
        out = ref_single_layer_attention(q, k, v, mask, 2, 2, 1)
        s = q @ k.transpose(-2, -1) / (8 ** 0.5)
        attn = torch.softmax(s, dim=-1)
        expected = attn @ v
        assert torch.allclose(out[0, 0], expected[0, 0], atol=1e-6)
