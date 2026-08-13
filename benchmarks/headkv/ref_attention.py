"""HeadKV Tensor-level reference:纯 PyTorch 最小语义实现。

用于验证 HeadReallocAttnBackend 的 full/comp 双路 attention 语义:
- full head:标准 causal attention(全历史)
- comp head:只允许访问 [0, sink) ∪ [L-recent, L)(sink + recent 窗口)

纯 torch,不依赖 SGLang/flashinfer。输入 q/k/v 已是每层投影后的张量。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))


def _window_mask(seq_len: int, sink: int, recent: int,
                 device: torch.device) -> torch.Tensor:
    """comp head 可见性矩阵:[i, j] = 位置 i 能否看到位置 j。

    j < sink 或 j >= seq_len - recent(且 j <= i,因果)。
    """
    causal = _causal_mask(seq_len, device)  # [L, L]
    allowed = torch.zeros(seq_len, dtype=torch.bool, device=device)
    allowed[:sink] = True
    if recent > 0:
        allowed[seq_len - recent:] = True
    return causal & allowed.unsqueeze(0)  # 每行 mask 相同列集合


def ref_single_layer_attention(
    q: torch.Tensor,      # [T, num_heads, head_dim] 或 [1, num_heads, head_dim]
    k: torch.Tensor,      # [L, num_heads, head_dim]
    v: torch.Tensor,      # [L, num_heads, head_dim]
    head_mask: torch.Tensor,  # [num_heads] bool,True=full
    sink: int,
    recent: int,
    num_kv_groups: int = 1,  # GQA:每 KV head 对应 Q head 数
    scale: float = None,
) -> torch.Tensor:
    """对一批 query token 计算 reference attention。

    支持 MHA(num_kv_groups=1)与 GQA;comp head 用窗口 mask。
    """
    T, num_q_heads, head_dim = q.shape
    L = k.shape[0]
    num_kv_heads = k.shape[1]
    assert num_q_heads == num_kv_heads * num_kv_groups

    if scale is None:
        scale = head_dim ** -0.5

    # 因果 mask:[T, L](query i 只能看 j <= i)
    causal = torch.tril(
        torch.ones(T, L, dtype=torch.bool, device=q.device)
    )
    # 窗口允许列集合:[L]
    window_allow = torch.zeros(L, dtype=torch.bool, device=q.device)
    window_allow[:sink] = True
    if recent > 0:
        window_allow[L - recent:] = True
    window_allow_t = window_allow.unsqueeze(0).expand(T, L)  # [T, L]

    # GQA 展开 KV(每 KV head 复制 num_kv_groups 份)→ [num_q_heads]
    k_exp = k.repeat_interleave(num_kv_groups, dim=1)  # [L, num_q_heads, dim]
    v_exp = v.repeat_interleave(num_kv_groups, dim=1)

    scores = torch.einsum("thd,lhd->htl", q, k_exp) * scale  # [H, T, L]

    out = torch.empty_like(q)
    for h in range(num_q_heads):
        if head_mask[h]:
            mask = causal  # full head
        else:
            mask = causal & window_allow_t  # comp head
        scores_h = scores[h].masked_fill(~mask, float("-inf"))
        attn = F.softmax(scores_h, dim=-1)  # [T, L]
        out[:, h, :] = torch.einsum("tl,ld->td", attn, v_exp[:, h, :])
    return out


def ref_sequence(
    q_list, k_list, v_list,
    head_mask, sink, recent, num_kv_groups=1,
):
    """逐 token 模拟(extend 后 decode):每个 q 是当前 token 的 query。

    k_list/v_list 是该 token 的 KV 投影(全部历史)。
    返回每个 token 的输出(用于 decode 步对比)。
    """
    outs = []
    L = len(k_list)
    for t in range(L):
        q_t = q_list[t].unsqueeze(0)  # [1, H, dim]
        k_all = torch.stack(k_list[: t + 1], dim=0)  # [t+1, H, dim]
        v_all = torch.stack(v_list[: t + 1], dim=0)
        o = ref_single_layer_attention(
            q_t, k_all, v_all, head_mask, sink, recent, num_kv_groups
        )
        outs.append(o[0])
    return torch.stack(outs)
