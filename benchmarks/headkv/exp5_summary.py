"""E1-E5 实验汇总: 生成 results/exp5_quality.csv + 汇总 json。

数据来源:
- E5 NIAH: exp5_official_niah_{full,duo_attn}.json(官方 HF)
          + s1_quality.json / niah_mini.json(HeadKV SGLang 既有)
- E5 LongBench: 官方 pred 打分 + exp5_headkv_* (官方协议, 30 条)
"""
import csv
import json

OUT = "/home/lixinze/HeadKV-SGLang/results/exp5_quality.csv"
ART = "/home/lixinze/HeadKV-SGLang/artifacts"

# ---- LongBench F1(官方 eval.py 打分结果, 30 条官方协议) ----
lb = {
    "narrativeqa": {"full_hf": 31.49, "fullkv_sglang": 31.49,
                    "duo_hf": 28.26, "headkv_sglang": 32.48},
    "2wikimqa": {"full_hf": 21.35, "fullkv_sglang": 20.38,
                 "duo_hf": 20.98, "headkv_sglang": 18.89},
}

# ---- NIAH 4K(9 题 magic 协议) ----
niah = {
    "full_hf": 9, "duo_hf": 9, "fullkv_sglang": 9, "headkv_sglang": 9,
}

rows = []
for task in ["narrativeqa", "2wikimqa"]:
    for k, v in lb[task].items():
        rows.append({"task": task, "system": k, "metric": "F1", "value": v})
for k, v in niah.items():
    rows.append({"task": "niah_4k", "system": k, "metric": "hit/9", "value": v})

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["task", "system", "metric", "value"])
    w.writeheader()
    w.writerows(rows)

summary = {
    "niah_4k_hits": niah,
    "longbench_f1_30": lb,
}
with open(f"{ART}/exp5_quality_summary.json", "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("=== NIAH 4K (hit/9) ===")
for k, v in niah.items():
    print(f"  {k:16s}: {v}/9")
print("=== LongBench F1 (30 条, 官方协议) ===")
for task in ["narrativeqa", "2wikimqa"]:
    r = lb[task]
    print(f"  {task:12s}: Full(HF)={r['full_hf']}  FullKV(SG)={r['fullkv_sglang']}  "
          f"Duo(HF)={r['duo_hf']}  HeadKV(SG)={r['headkv_sglang']}")
print(f"saved -> {OUT}")
