"""Phase 2 Gate 2 验证:真实官方 pattern 的 summarize 输出 + ratio 数据。

用途:
1. 验证 DuoAttentionPolicy 端到端可用
2. 记录官方 threshold=0.5 pattern 的 effective full ratio(设计文档 §11 待确认 4)
3. 验证确定性(两次运行 mask 一致)
"""
import json
from pathlib import Path

import torch

from sglang.srt.headkv.config import HeadKVConfig
from sglang.srt.headkv.duo_policy import DuoAttentionPolicy

PATTERN = str(
    Path(__file__).resolve().parent / "data" / "meta-llama-3.1-8b-instruct"
)


class ModelConfig:
    num_hidden_layers = 32
    num_attention_heads = 32
    num_key_value_heads = 8

    def get_num_kv_heads(self, tp_size=1):
        return self.num_key_value_heads // tp_size


def main():
    # 官方语义:threshold=0.5(不指定 ratio/threshold → config.json threshold)
    cfg = HeadKVConfig(enable=True, policy="duo", pattern_path=PATTERN,
                       max_running_requests=32)
    cfg.validate()
    pol = DuoAttentionPolicy(cfg)
    m1 = pol.load_global_kv_mask(ModelConfig())
    s = pol.summarize()
    print("=== threshold=0.5(官方默认)===")
    print(json.dumps(s, indent=2, ensure_ascii=False))

    # 确定性验证
    m2 = DuoAttentionPolicy(cfg).load_global_kv_mask(ModelConfig())
    print(f"deterministic: {torch.equal(m1, m2)}")

    # ratio 扫描(Experiment A 预演)
    print("=== full_head_ratio 扫描 ===")
    for ratio in (0.25, 0.50, 0.75):
        c = HeadKVConfig(enable=True, policy="duo", pattern_path=PATTERN,
                         full_head_ratio=ratio, max_running_requests=32)
        p = DuoAttentionPolicy(c)
        p.load_global_kv_mask(ModelConfig())
        ss = p.summarize()
        print(f"  ratio={ratio}: full={ss['full_heads']}/{ss['mask_shape'][0]*ss['mask_shape'][1]} "
              f"effective={ss['effective_full_ratio']}")


if __name__ == "__main__":
    main()
