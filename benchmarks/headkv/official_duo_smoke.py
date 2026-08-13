"""Official DuoAttention smoke(Phase 0 Gate 0 第二部分):同一模型+pattern 生成。

用官方 duo-attention-ref 的 eval patch + HF transformers(纯 oracle,不做系统性能)。
用法:
    PYTHONPATH=~/duo-attention-ref CUDA_VISIBLE_DEVICES=1 python official_duo_smoke.py \
        --model ~/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct \
        --pattern ~/duo-attention-ref/attn_patterns/Meta-Llama-3.1-8B-Instruct/lr=0.02-reg=0.05-ctx=1000_128000-multi_passkey10 \
        --out artifacts/official_duo_smoke.json
"""
import argparse
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from duo_attn.patch import enable_duo_attention_eval, load_full_attention_heads

MODEL_REVISION_CHECK = "Meta-Llama-3.1-8B-Instruct"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pattern", required=True, help="pattern 目录(含 tsv + config.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="What is the capital of France? Answer in one word.")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    args = ap.parse_args()

    # 1. 读取 pattern config 的 sink/recent(官方部署语义)
    with open(os.path.join(args.pattern, "config.json")) as f:
        cfg = json.load(f)
    sink = cfg.get("deploy_sink_size") or cfg.get("sink_size")
    recent = cfg.get("deploy_recent_size") or cfg.get("recent_size")
    assert sink is not None and recent is not None, "pattern config.json 缺 sink/recent"
    print(f"[smoke] pattern config: sink={sink} recent={recent} threshold={cfg.get('threshold')}")

    # 2. 加载 pattern(官方 loader,内部 clip)
    heads = load_full_attention_heads(args.pattern, filename="full_attention_heads.tsv")
    print(f"[smoke] pattern shape={heads.shape} min={heads.min():.4f} max={heads.max():.4f}")

    # 3. 加载模型(本地,bf16,GPU 1)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda:1",
        trust_remote_code=True,
    )
    model.eval()

    # 4. 启用 DuoAttention eval patch
    enable_duo_attention_eval(model, heads, sink_size=sink, recent_size=recent)

    # 5. 单请求生成(temperature=0)
    msgs = [{"role": "user", "content": args.prompt}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to("cuda:1")
    t0 = time.time()
    with torch.inference_mode():
        out_ids = model.generate(
            ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            use_cache=True,
        )
    dt = time.time() - t0
    gen = out_ids[0][ids.shape[1]:]
    text = tok.decode(gen, skip_special_tokens=True)

    result = {
        "model": args.model,
        "pattern": args.pattern,
        "pattern_shape": list(heads.shape),
        "pattern_min_max": [float(heads.min()), float(heads.max())],
        "sink": sink, "recent": recent,
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "elapsed_s": round(dt, 3),
        "output": text,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps({"elapsed_s": round(dt, 3), "output": text}, ensure_ascii=False))
    print(f"[smoke] saved -> {args.out}")


if __name__ == "__main__":
    main()
