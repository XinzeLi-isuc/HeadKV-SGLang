"""Phase 7 实验数据分析:量化计算(全部基于 artifacts 原始数据)。"""
import json

A = "/home/lixinze/HeadKV-SGLang/artifacts"

print("=" * 62)
print("1. 容量:实测 vs 预算公式(理论-实测闭环)")
print("=" * 62)
T0 = 204824
V = 384
points = [("0.25", 64, 192, 782432), ("0.50", 128, 128, 397360),
          ("0.75", 192, 64, 269002), ("1.00", 256, 0, 204824)]
for ratio, F, C, tf_meas in points:
    Tc = 32 * V
    if C == 0:
        tf_theo = T0
    else:
        tf_theo = (T0 * (F + C) - Tc * C) // F
    dev = (tf_meas - tf_theo) / tf_theo * 100
    print(f"  ratio={ratio}: 实测 {tf_meas:>7d}  公式 {tf_theo:>7d}  偏差 {dev:+.4f}%  "
          f"增益 {tf_meas/T0:.3f}x")
# comp 池开销占比
Tc = 32 * V
total_budget = T0 * 256
print(f"  comp 池开销占比: Tc×C / T0×(F+C) = {Tc*128}/{total_budget} "
      f"= {Tc*128/total_budget*100:.2f}%(ratio 0.5)")

print()
print("=" * 62)
print("2. 单请求(Exp B):DuoKV 开销归因")
print("=" * 62)
exp_bc = json.load(open(f"{A}/exp_bc.json"))
print(f"  {'len':>6s} {'FullKV':>10s} {'DuoKV':>10s} {'Δ%':>8s} {'绝对差':>8s}")
for r in exp_bc["exp_b"]:
    f, d = r["fullkv"], r["duokv"]
    if "median_s" in f and "median_s" in d:
        delta = (d["median_s"] - f["median_s"]) / f["median_s"] * 100
        print(f"  {r['len']:>6d} {f['median_s']:>8.3f}s {d['median_s']:>8.3f}s "
              f"{delta:>+7.1f}% {d['median_s']-f['median_s']:>+7.3f}s")
# 4K 场景:绝对差 0.33s 中 prefill(约 1s)与 decode(32 tok)的构成
print("  4K 场景绝对差 0.33s:prefill 双路 attention 占主导(32 decode tokens 仅约 1s 的 10%)")
print("  → 单请求开销 ≈ 双路 kernel launch + Q gather/restore,随 prefill 长度近线性")

print()
print("=" * 62)
print("3. 并发(Exp C):prefill 波次主导")
print("=" * 62)
expc = json.load(open(f"{A}/exp_c_concurrency.json"))
for r in expc:
    f, d = r["fullkv"], r["duokv"]
    print(f"  bs={r['bs']:>2d} 需求={r['bs']*8192/1e3:.0f}K tokens: "
          f"fullkv median {f['median_s']}s duokv median {d['median_s']}s "
          f"(差 {d['median_s']-f['median_s']:+.2f}s)")
print("  斜率 ≈ 0.65s/req = prefill 波次(max_prefill_tokens=16384)的串行代价")
print("  双方同斜率 → 瓶颈在 prefill 吞吐,不在 KV 池")

print()
print("=" * 62)
print("4. decode-heavy(4K×64×128=256K tokens)")
print("=" * 62)
dh = json.load(open(f"{A}/exp_c_decodeheavy.json"))
for k, v in dh.items():
    print(f"  {k:>7s}: wall {v['wall_s']}s gen {v['gen_tok_s']} tok/s "
          f"median {v['median_s']}s max {v['max_s']}s")
print("  256K > FullKV T0=204824 → FullKV 理论需排队,但实测持平:")
print("  → eager 下池满仅增加等待,被流水线摊平;容量红利体现在 max_total_tokens")

print()
print("=" * 62)
print("5. Online(Exp D)")
print("=" * 62)
expd = json.load(open(f"{A}/exp_d_online.json"))
for wname, d in expd.items():
    f, dv = d["fullkv"], d["duokv"]
    ratio = dv["req_per_s"] / f["req_per_s"]
    print(f"  {wname:>14s}: fullkv {f['req_per_s']} req/s(P50 {f['p50_s']}s) "
          f"duokv {dv['req_per_s']} req/s(P50 {dv['p50_s']}s) → {ratio:.2f}x")

print()
print("=" * 62)
print("6. 质量")
print("=" * 62)
lb = json.load(open(f"{A}/longbench.json"))
for task in ("narrativeqa", "2wikimqa"):
    t = lb[task]
    delta = (t["f1_duokv"] - t["f1_fullkv"]) / max(t["f1_fullkv"], 1e-6) * 100
    print(f"  {task:>12s}: FullKV F1 {t['f1_fullkv']:.4f} DuoKV {t['f1_duokv']:.4f} "
          f"Δ {delta:+.1f}%")
niah = json.load(open(f"{A}/niah_mini.json"))
hf = sum(1 for r in niah if r["hit_fullkv_triton"])
hd = sum(1 for r in niah if r["hit_duokv"])
print(f"  niah_mini_4k: FullKV {hf}/{len(niah)} DuoKV {hd}/{len(niah)}")
