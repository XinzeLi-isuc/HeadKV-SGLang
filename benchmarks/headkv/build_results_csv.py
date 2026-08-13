"""Phase 8:汇总 results/*.csv(从 artifacts 提取关键数据)。"""
import csv
import json
import os

os.makedirs("/home/lixinze/HeadKV-SGLang/results", exist_ok=True)
A = "/home/lixinze/HeadKV-SGLang/artifacts"

# capacity.csv
with open(f"{A}/capacity_sweep.csv") as f:
    rows = [r.split(",") for r in f.read().strip().splitlines()[1:]]
rows.append(["0.50", "397360", "397360"])  # Phase 3 实测(ratio 0.5)
rows.append(["1.00", "204824", "204824"])  # FullKV 等效点
with open("/home/lixinze/HeadKV-SGLang/results/capacity.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["full_head_ratio", "Tf", "max_total_num_tokens"])
    w.writerows(rows)
print("capacity.csv:", len(rows), "rows")

# throughput.csv(Exp B)
exp_bc = json.load(open(f"{A}/exp_bc.json"))
with open("/home/lixinze/HeadKV-SGLang/results/throughput.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["ctx_len", "system", "median_e2e_s", "tok_per_s"])
    for r in exp_bc["exp_b"]:
        for name in ("fullkv", "duokv"):
            v = r[name]
            w.writerow([r["len"], name, v.get("median_s"), v.get("tok_per_s")])
print("throughput.csv: done")

# online_serving.csv(Exp D)
exp_d = json.load(open(f"{A}/exp_d_online.json"))
with open("/home/lixinze/HeadKV-SGLang/results/online_serving.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["workload", "system", "req_per_s", "p50_s", "p95_s", "max_s", "n"])
    for wname, d in exp_d.items():
        for name in ("fullkv", "duokv"):
            v = d[name]
            w.writerow([wname, name, v.get("req_per_s"), v.get("p50_s"),
                        v.get("p95_s"), v.get("max_s"), v.get("n")])
print("online_serving.csv: done")

# quality.csv(Exp E)
lb = json.load(open(f"{A}/longbench.json"))
with open("/home/lixinze/HeadKV-SGLang/results/quality.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["task", "metric", "fullkv", "duokv"])
    for task in ("narrativeqa", "2wikimqa"):
        w.writerow([task, "F1", lb[task]["f1_fullkv"], lb[task]["f1_duokv"]])
    # NIAH mini
    niah = json.load(open(f"{A}/niah_mini.json"))
    n = len(niah)
    hf = sum(1 for r in niah if r["hit_fullkv_triton"])
    hd = sum(1 for r in niah if r["hit_duokv"])
    w.writerow(["niah_mini_4k", "hit_rate", hf / n, hd / n])
print("quality.csv: done")
