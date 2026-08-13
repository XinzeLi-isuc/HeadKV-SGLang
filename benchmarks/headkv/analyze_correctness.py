"""分析 E2E correctness 结果:分叉点位置 + 首 token 一致率。"""
import json

with open("/home/lixinze/HeadKV-SGLang/artifacts/correctness_e2e.json") as f:
    data = json.load(f)

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(
    "/home/lixinze/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct"
)

print(f"total={data['total']} exact={data['exact_match']} ({data['exact_ratio']})")
print(f"{'id':6s} {'type':20s} {'first-div':>9s} {'first-tok':>9s} {'prompt_len':>10s}")
n_first = 0
for r in data["results"]:
    ids_f = tok(r["fullkv"])["input_ids"]
    ids_d = tok(r["duokv"])["input_ids"]
    n = min(len(ids_f), len(ids_d))
    first_div = n
    for i in range(n):
        if ids_f[i] != ids_d[i]:
            first_div = i
            break
    first_tok_match = ids_f[0] == ids_d[0]
    n_first += int(first_tok_match)
    print(f"{r['id']:6s} {r['type']:20s} {first_div:>9d} {str(first_tok_match):>9s} "
          f"{len(ids_f):>10d}")

print(f"\nfirst-token match: {n_first}/{data['total']} = {n_first/data['total']:.2f}")
