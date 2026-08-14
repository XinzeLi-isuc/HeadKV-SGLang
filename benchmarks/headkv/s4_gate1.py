"""S4 Gate 1:HeadPolicy 在 current SGLang main 环境下的可加载性。

验证 headkv 包(纯算法)与 current main ModelConfig 接口的兼容性:
- current 风格 ModelConfig(get_num_kv_heads(tp, dcp=1) 签名)
- 真实 Duo pattern + RLKV adapter 加载
- budget/partition 计算
"""
import sys

sys.path.insert(0, "/home/lixinze/sglang-main/python/sglang/srt")

import torch  # noqa: E402

from sglang.srt.headkv.budget import compute  # noqa: E402
from sglang.srt.headkv.config import HeadKVConfig  # noqa: E402
from sglang.srt.headkv.partition import to_tp_local  # noqa: E402
from sglang.srt.headkv.policy import HeadPolicy  # noqa: E402

DUO = "/home/lixinze/duo-attention-ref/attn_patterns/Meta-Llama-3.1-8B-Instruct/lr=0.02-reg=0.05-ctx=1000_128000-multi_passkey10"
RLKV = "/home/lixinze/rlkv/head_dist/rlkv/Llama-3.1-8B-R1/llama_lr1e-2_ep2_bs32_reg1e-3_tau0.5"


class CurrentStyleModelConfig:
    """按 current main ModelConfig 的公开接口构造(鸭子类型)。"""

    num_hidden_layers = 32
    num_attention_heads = 32
    num_key_value_heads = 8
    head_dim = 128
    context_len = 131072

    def get_num_kv_heads(self, tensor_parallel_size: int, dcp_size: int = 1) -> int:
        return self.num_key_value_heads // tensor_parallel_size

    def get_num_attention_heads(self, tensor_parallel_size: int) -> int:
        return self.num_attention_heads // tensor_parallel_size


def main():
    mc = CurrentStyleModelConfig()
    print("[Gate1] current 风格 ModelConfig 构造 OK")

    # Duo policy 加载
    cfg_d = HeadKVConfig(
        enable=True, policy="duo", pattern_path=DUO,
        full_head_ratio=0.5, sink_size=128, recent_size=256,
        max_running_requests=32,
    )
    cfg_d.validate()
    p_d = HeadPolicy.create(cfg_d)
    mask_d = p_d.load_global_kv_mask(mc)
    assert mask_d.shape == (32, 8), mask_d.shape
    print(f"[Gate1] DuoAttentionPolicy 加载 OK: shape={mask_d.shape} "
          f"full={int(mask_d.sum())} sink={p_d.sink_size()} recent={p_d.recent_size()}")

    # RLKV policy 加载
    cfg_r = HeadKVConfig(
        enable=True, policy="rlkv", pattern_path=RLKV,
        sparsity=0.5, sink_size=16, recent_size=32,
        max_running_requests=32,
    )
    cfg_r.validate()
    p_r = HeadPolicy.create(cfg_r)
    mask_r = p_r.load_global_kv_mask(mc)
    assert mask_r.shape == (32, 8), mask_r.shape
    print(f"[Gate1] RLKVPolicy 加载 OK: shape={mask_r.shape} "
          f"full={int(mask_r.sum())}")

    # budget + partition(TP=1)
    tp_mask = to_tp_local(mask_d, 0, 1, mc.get_num_kv_heads(1))
    total_full = sum(int(m.sum().item()) for m in tp_mask.values())
    total_comp = 32 * 8 - total_full
    b = compute(T0=204824, F=total_full, C=total_comp, R=32, V=384)
    print(f"[Gate1] budget OK: F={total_full} C={total_comp} "
          f"Tf={b.Tf} Tc={b.Tc} gain={b.predicted_gain:.3f}x")

    print("\n[Gate1] PASS: HeadPolicy 在 current main 接口下可加载")
    print("(pool/allocator/backend 接入为 Gate 2-3)")


if __name__ == "__main__":
    main()
