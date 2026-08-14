"""HeadKV dual-pool for current SGLang main.

Port of RLKV v0.5.2 HeadReallocKVPool / HeadReallocAllocator to the
current SGLang main interfaces (KVCache ABC + BaseTokenToKVPoolAllocator).
Semantics unchanged: per-layer full/comp buffers, per-request comp chunk,
full→comp location mapping.

Baseline: rlkv-sglang-v0.5.2@973b5e41 (memory_pool.py L892 / allocator.py L290).
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import torch

from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import KVCache

logger = logging.getLogger(__name__)


class HeadReallocKVPool(KVCache):
    """KV cache with separate pools for full and compressed attention heads.

    Full heads get a pool for all tokens (full context).
    Compressed heads get a separate pool (for sink + recent window tokens).
    Each pool has per-layer varying head counts based on head masks.
    """

    def __init__(
        self,
        size_full: int,
        size_comp: int,
        head_masks: Dict[int, torch.Tensor],
        head_dim: int,
        layer_num: int,
        dtype: torch.dtype,
        device: str,
        enable_memory_saver: bool,
    ):
        super().__init__(
            size_full, 1, dtype, layer_num, device, enable_memory_saver
        )
        self.size_full = size_full
        self.size_comp = size_comp
        self.head_dim = head_dim

        # Per-layer head indices for splitting incoming KV
        self.full_head_indices: Dict[int, torch.Tensor] = {}
        self.comp_head_indices: Dict[int, torch.Tensor] = {}

        for layer_id, mask in head_masks.items():
            self.full_head_indices[layer_id] = torch.where(mask == 1)[0].to(device)
            self.comp_head_indices[layer_id] = torch.where(mask == 0)[0].to(device)

        # Create per-layer buffers with varying head counts
        self.full_k_buffer = []
        self.full_v_buffer = []
        self.comp_k_buffer = []
        self.comp_v_buffer = []

        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            for l in range(layer_num):
                n_full = len(self.full_head_indices.get(l, []))
                n_comp = len(self.comp_head_indices.get(l, []))

                self.full_k_buffer.append(
                    torch.zeros(
                        size_full + 1, n_full, head_dim,
                        dtype=self.store_dtype, device=device,
                    )
                )
                self.full_v_buffer.append(
                    torch.zeros(
                        size_full + 1, n_full, head_dim,
                        dtype=self.store_dtype, device=device,
                    )
                )
                self.comp_k_buffer.append(
                    torch.zeros(
                        size_comp + 1, n_comp, head_dim,
                        dtype=self.store_dtype, device=device,
                    )
                )
                self.comp_v_buffer.append(
                    torch.zeros(
                        size_comp + 1, n_comp, head_dim,
                        dtype=self.store_dtype, device=device,
                    )
                )

        # Mapping from full pool locations to comp pool locations (set by allocator)
        self.full_to_comp_mapping: Optional[torch.Tensor] = None

        self._finalize_allocation_log(size_full)

    def get_kv_size_bytes(self):
        k_size = sum(b.nelement() * b.element_size() for b in self.full_k_buffer)
        k_size += sum(b.nelement() * b.element_size() for b in self.comp_k_buffer)
        v_size = sum(b.nelement() * b.element_size() for b in self.full_v_buffer)
        v_size += sum(b.nelement() * b.element_size() for b in self.comp_v_buffer)
        return k_size, v_size

    def translate_loc_full_to_comp(self, loc: torch.Tensor) -> torch.Tensor:
        assert self.full_to_comp_mapping is not None
        return self.full_to_comp_mapping[loc].to(torch.int32)

    def get_key_buffer(self, layer_id: int):
        if self.store_dtype != self.dtype:
            return self.full_k_buffer[layer_id].view(self.dtype)
        return self.full_k_buffer[layer_id]

    def get_value_buffer(self, layer_id: int):
        if self.store_dtype != self.dtype:
            return self.full_v_buffer[layer_id].view(self.dtype)
        return self.full_v_buffer[layer_id]

    def get_kv_buffer(self, layer_id: int):
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def get_comp_key_buffer(self, layer_id: int):
        if self.store_dtype != self.dtype:
            return self.comp_k_buffer[layer_id].view(self.dtype)
        return self.comp_k_buffer[layer_id]

    def get_comp_value_buffer(self, layer_id: int):
        if self.store_dtype != self.dtype:
            return self.comp_v_buffer[layer_id].view(self.dtype)
        return self.comp_v_buffer[layer_id]

    def set_kv_buffer(
        self,
        layer,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
    ):
        """Split incoming KV by layer head mask: full heads → full pool,
        comp heads → comp pool (via full_to_comp_mapping)."""
        if layer_id_override is not None:
            layer_id = layer_id_override
        else:
            layer_id = layer.layer_id

        full_locs = loc
        comp_locs = self.translate_loc_full_to_comp(loc)

        n_full = len(self.full_head_indices.get(layer_id, []))
        n_comp = len(self.comp_head_indices.get(layer_id, []))

        # current main passes cache_k/cache_v as [num_tokens, kv_heads, dim]
        # (v0.5.2 used [num_tokens, 1, kv_heads, dim]); support both.
        dim4 = cache_k.dim() == 4

        if n_full > 0:
            if dim4:
                full_k = cache_k[:, :, self.full_head_indices[layer_id], :]
                full_v = cache_v[:, :, self.full_head_indices[layer_id], :]
            else:
                full_k = cache_k[:, self.full_head_indices[layer_id], :]
                full_v = cache_v[:, self.full_head_indices[layer_id], :]
            full_k = full_k.contiguous().view(-1, n_full, self.head_dim)
            full_v = full_v.contiguous().view(-1, n_full, self.head_dim)
            self.full_k_buffer[layer_id][full_locs] = full_k
            self.full_v_buffer[layer_id][full_locs] = full_v

        if n_comp > 0:
            if dim4:
                comp_k = cache_k[:, :, self.comp_head_indices[layer_id], :]
                comp_v = cache_v[:, :, self.comp_head_indices[layer_id], :]
            else:
                comp_k = cache_k[:, self.comp_head_indices[layer_id], :]
                comp_v = cache_v[:, self.comp_head_indices[layer_id], :]
            comp_k = comp_k.contiguous().view(-1, n_comp, self.head_dim)
            comp_v = comp_v.contiguous().view(-1, n_comp, self.head_dim)
            self.comp_k_buffer[layer_id][comp_locs] = comp_k
            self.comp_v_buffer[layer_id][comp_locs] = comp_v


class HeadReallocAllocator(BaseTokenToKVPoolAllocator):
    """Allocator for head reallocation dual KV pool.

    Manages the full pool via standard alloc/free. Comp pool uses
    chunk-based allocation: each request gets a fixed window of comp
    slots (allocated by the attention backend on first extend,
    freed automatically when request tokens are freed).

    The full_to_comp_mapping is populated by the attention backend
    using circular addressing within each request's comp window.
    """

    def __init__(
        self,
        size_full: int,
        size_comp: int,
        dtype: torch.dtype,
        device: str,
        kvcache,
        need_sort: bool,
        window_size: int,
    ):
        super().__init__(size_full, 1, dtype, device, kvcache, need_sort)
        self._size_full = size_full
        self._size_comp = size_comp
        self.window_size = window_size

        self.full_allocator = TokenToKVPoolAllocator(
            size_full, dtype, device, kvcache, need_sort,
        )

        # Comp pool chunk management.
        # Each chunk is window_size contiguous comp slots.
        # comp_base = 1 + chunk_id * window_size.
        self.max_comp_chunks = size_comp // window_size if window_size > 0 else 0
        self._comp_free_chunks = None  # initialized in clear()

        # Mapping from full pool loc → comp pool loc.
        self.full_to_comp_mapping = torch.zeros(
            size_full + 2,
            dtype=torch.int64,
            device=device,
        )
        self.clear()
        self._kvcache.full_to_comp_mapping = self.full_to_comp_mapping

    def available_size(self):
        return self.full_allocator.available_size()

    def comp_chunks_available(self):
        return len(self._comp_free_chunks) if self._comp_free_chunks else 0

    def debug_print(self) -> str:
        return (
            f"#full-available: {self.full_allocator.available_size()}, "
            f"#comp-chunks: {self.comp_chunks_available()}/{self.max_comp_chunks}"
        )

    def alloc_comp_window(self) -> int:
        """Allocate a window-sized comp chunk. Returns comp_base (>0) or 0 if full."""
        if not self._comp_free_chunks:
            return 0
        chunk_id = self._comp_free_chunks.pop()
        return 1 + chunk_id * self.window_size

    def free_comp_window(self, comp_base: int):
        """Free a window-sized comp chunk."""
        if comp_base > 0 and self.window_size > 0:
            chunk_id = (comp_base - 1) // self.window_size
            if chunk_id < self.max_comp_chunks:
                self._comp_free_chunks.append(chunk_id)

    def alloc(self, need_size: int):
        """Allocate from full pool only. Comp is managed per-request."""
        return self.full_allocator.alloc(need_size)

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return
        if self.is_not_in_free_group:
            self.full_allocator.free(free_index)
            # Free comp chunks by deriving comp_base from mapping
            comp_indices = self.full_to_comp_mapping[free_index]
            non_zero = comp_indices[comp_indices > 0]
            if non_zero.numel() > 0:
                comp_bases = (
                    (non_zero - 1) // self.window_size * self.window_size + 1
                )
                unique_bases = torch.unique(comp_bases)
                for cb in unique_bases.tolist():
                    self.free_comp_window(cb)
            self.full_to_comp_mapping[free_index] = 0
        else:
            self.free_group.append(free_index)

    def clear(self):
        self.full_allocator.clear()
        self.full_to_comp_mapping.fill_(0)
        self._comp_free_chunks = list(range(self.max_comp_chunks))
        self.is_not_in_free_group = True
        self.free_group = []


from sglang.srt.mem_cache.memory_pool import GPU_MEMORY_TYPE_KV_CACHE  # noqa: E402
