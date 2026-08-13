"""调试 test_headkv_attention 的 extend mapping 实际行为。"""
import sys

import torch

sys.path.insert(0, "/home/lixinze/rlkv/sglang/test/srt/headkv")
from test_headkv_attention import _make_env, _extend, SINK, RECENT, V  # noqa: E402

_, pool, allocator, backend = _make_env([1, 0])
print("max_comp_chunks:", allocator.max_comp_chunks)
print("comp_free_chunks:", allocator._comp_free_chunks)
print("window_size:", allocator.window_size)

full_locs, comp_base = _extend(None, pool, allocator, backend, 0, 0, 60)
print("full_locs[:5]:", full_locs[:5].tolist(), "len:", len(full_locs))
print("comp_base(测试拿到的):", comp_base)
mapping = pool.full_to_comp_mapping[full_locs]
print("mapping[:10]:", mapping[:10].tolist())
print("mapping[44:60]:", mapping[44:60].tolist())
print("req_to_token[0,0]:", backend.req_to_token[0, 0].item())
print("mapping[1](position0 的映射):", pool.full_to_comp_mapping[1].item())
