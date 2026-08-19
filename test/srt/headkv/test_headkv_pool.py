"""HeadReallocKVPool / HeadReallocAllocator 物理测试(纯 CPU,current main 移植版)。

关键覆盖:allocator 不变量 —— Full slots 与 Compact chunks 在请求结束后
全部恢复、full_to_comp_mapping 清零、comp 池耗尽 fail-fast。
基线:rlkv-sglang-v0.5.2 test/srt/headkv/test_headkv_pool.py(v0.5.2 用
allocator 物理测试;current-main 用 headkv_pool.py 的对应实现)。
"""
import pytest
import torch

from test_headkv_attention import (  # noqa: E402
    HEAD_DIM,
    KV_HEADS,
    N_LAYERS,
    RECENT,
    SINK,
    V,
    FakeArgs,
    FakeModelConfig,
    FakeReqToTokenPool,
    FakeRunner,
    _decode,
    _extend,
)

from sglang.srt.headkv.partition import to_tp_local  # noqa: E402
from sglang.srt.layers.attention.headkv_backend import (  # noqa: E402
    HeadReallocAttnBackend,
)
from sglang.srt.mem_cache.headkv_pool import (  # noqa: E402
    HeadReallocAllocator,
    HeadReallocKVPool,
)

SIZE_FULL, SIZE_COMP = 2048, 4 * V


def _make_env(mask_layer):
    masks = to_tp_local(torch.tensor([mask_layer], dtype=torch.bool), 0, 1, KV_HEADS)
    runner = FakeRunner(masks)
    pool = HeadReallocKVPool(
        size_full=SIZE_FULL, size_comp=SIZE_COMP, head_masks=masks,
        head_dim=HEAD_DIM, layer_num=N_LAYERS, dtype=torch.float32,
        device="cpu", enable_memory_saver=False,
    )
    runner.token_to_kv_pool = pool
    allocator = HeadReallocAllocator(
        SIZE_FULL, SIZE_COMP, torch.float32, "cpu", pool,
        need_sort=False, window_size=V,
    )
    runner.token_to_kv_pool_allocator = allocator
    backend = HeadReallocAttnBackend(runner, masks)
    backend._allocator_ref = allocator
    return runner, pool, allocator, backend


class TestPoolShapes:
    def test_pool_tensor_shapes(self):
        _, pool, _, _ = _make_env([1, 0])
        assert pool.full_k_buffer[0].shape == (SIZE_FULL + 1, 1, HEAD_DIM)
        assert pool.comp_k_buffer[0].shape == (SIZE_COMP + 1, 1, HEAD_DIM)
        assert pool.full_v_buffer[0].shape == (SIZE_FULL + 1, 1, HEAD_DIM)
        assert pool.comp_v_buffer[0].shape == (SIZE_COMP + 1, 1, HEAD_DIM)

    def test_pool_tensor_shapes_all_comp(self):
        _, pool, _, _ = _make_env([0, 0])
        assert pool.full_k_buffer[0].shape == (SIZE_FULL + 1, 0, HEAD_DIM)
        assert pool.comp_k_buffer[0].shape == (SIZE_COMP + 1, 2, HEAD_DIM)

    def test_no_full_history_copy(self):
        _, pool, _, _ = _make_env([1, 0])
        for b in (pool.full_k_buffer, pool.full_v_buffer,
                  pool.comp_k_buffer, pool.comp_v_buffer):
            for t in b:
                assert t.shape[1] <= KV_HEADS

    def test_pool_byte_accounting(self):
        _, pool, _, _ = _make_env([1, 0])
        k_bytes, v_bytes = pool.get_kv_size_bytes()
        expected_k = sum(
            t.nelement() * t.element_size()
            for b in (pool.full_k_buffer, pool.comp_k_buffer) for t in b
        )
        expected_v = sum(
            t.nelement() * t.element_size()
            for b in (pool.full_v_buffer, pool.comp_v_buffer) for t in b
        )
        assert k_bytes == expected_k
        assert v_bytes == expected_v
        per_head = HEAD_DIM * 4  # float32
        assert k_bytes == (SIZE_FULL + 1) * 1 * per_head + (SIZE_COMP + 1) * 1 * per_head


class TestAllocator:
    def test_comp_chunk_count(self):
        _, _, allocator, _ = _make_env([1, 0])
        assert allocator.max_comp_chunks == SIZE_COMP // V == 4
        assert allocator.comp_chunks_available() == 4

    def test_comp_window_alloc_free_cycle(self):
        _, _, allocator, _ = _make_env([1, 0])
        bases = [allocator.alloc_comp_window() for _ in range(4)]
        assert sorted(bases) == [1, 25, 49, 73]
        assert allocator.comp_chunks_available() == 0
        assert allocator.alloc_comp_window() == 0  # 耗尽
        allocator.free_comp_window(bases[0])
        assert allocator.comp_chunks_available() == 1
        assert allocator.alloc_comp_window() == bases[0]

    def test_free_derives_comp_base_once(self):
        """同一 chunk 多 loc 批量释放只 free 一次。"""
        _, pool, allocator, backend = _make_env([1, 0])
        full_locs, comp_base = _extend(None, pool, allocator, backend, 0, 0, 30)
        assert allocator.comp_chunks_available() == 3  # 已用 1 个
        allocator.free(full_locs)
        assert allocator.comp_chunks_available() == 4  # 只归还一次
        assert (pool.full_to_comp_mapping[full_locs] == 0).all()  # mapping 清零

    def test_free_group_flush(self):
        _, pool, allocator, backend = _make_env([1, 0])
        full_locs, _ = _extend(None, pool, allocator, backend, 0, 0, 20)
        allocator.free_group_begin()
        allocator.free(full_locs)
        assert allocator.comp_chunks_available() == 3  # 延迟,未归还
        allocator.free_group_end()
        assert allocator.comp_chunks_available() == 4

    def test_all_full_degenerate_path(self):
        _, pool, allocator, _ = _make_env([1, 1])
        assert allocator.max_comp_chunks == SIZE_COMP // V
        full_locs = allocator.alloc(10)
        assert len(full_locs) == 10

    def test_allocator_restores_initial_state(self):
        """100 次 alloc/free 循环后恢复初始态。"""
        _, pool, allocator, backend = _make_env([1, 0])
        for i in range(100):
            full_locs, _ = _extend(None, pool, allocator, backend, 0, 0, 10)
            allocator.free(full_locs)
        assert allocator.available_size() == SIZE_FULL
        assert allocator.comp_chunks_available() == allocator.max_comp_chunks
        assert (pool.full_to_comp_mapping == 0).all()


class TestCompExhaustion:
    def test_comp_pool_exhaustion_fails_fast(self):
        """comp 池耗尽必须显式报错,不能静默写 dummy。"""
        from sglang.srt.headkv.partition import to_tp_local as _tpl

        masks = _tpl(torch.tensor([[1, 0]], dtype=torch.bool), 0, 1, KV_HEADS)
        runner = FakeRunner(masks)
        pool = HeadReallocKVPool(
            size_full=512, size_comp=1 * V, head_masks=masks,
            head_dim=HEAD_DIM, layer_num=N_LAYERS, dtype=torch.float32,
            device="cpu", enable_memory_saver=False,
        )
        runner.token_to_kv_pool = pool
        allocator = HeadReallocAllocator(
            512, 1 * V, torch.float32, "cpu", pool, need_sort=False, window_size=V,
        )
        runner.token_to_kv_pool_allocator = allocator
        backend = HeadReallocAttnBackend(runner, masks)
        backend._allocator_ref = allocator

        full_locs0 = allocator.alloc(10)
        runner.req_to_token_pool.req_to_token[0, :10] = torch.tensor(full_locs0)
        backend._update_comp_mapping_extend(pool, _FB(0, 0, 10))
        assert allocator.comp_chunks_available() == 0

        full_locs1 = allocator.alloc(10)
        runner.req_to_token_pool.req_to_token[1, :10] = torch.tensor(full_locs1)
        with pytest.raises(RuntimeError, match="comp"):
            backend._update_comp_mapping_extend(pool, _FB(1, 0, 10))


class _FB:
    def __init__(self, req_idx, prefix_len, seq_len):
        self.batch_size = 1
        self.req_pool_indices = torch.tensor([req_idx])
        self.extend_prefix_lens = torch.tensor([prefix_len])
        self.seq_lens = torch.tensor([seq_len])
