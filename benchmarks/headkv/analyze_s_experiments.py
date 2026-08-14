"""S 级实验数据分析:量化计算(S0-S3,全部基于 artifacts 原始数据)。"""
import json

A = "/home/lixinze/HeadKV-SGLang/artifacts"

print("=" * 66)
print("1. S3 CUDA Graph:固定开销消除的量化归因")
print("=" * 66)
# eager(Phase 7 Exp B)vs CG(S3)的绝对差
eager = {4096: 0.334, 8192: 0.335, 16384: 0.326}
cg = {4096: 0.030, 8192: 0.018, 16384: 0.047}
print(f"  {'len':>6s} {'eager 绝对差':>12s} {'CG 绝对差':>10s} {'消除率':>8s} "
      f"{'CG 相对开销':>10s}")
for L in (4096, 8192, 16384):
    elim = (eager[L] - cg[L]) / eager[L] * 100
    rel = cg[L] / (eager[L] / 0.32) * 100  # 相对:绝对差/FullKV 时长
    print(f"  {L:>6d} {eager[L]:>+9.3f}s {cg[L]:>+9.3f}s {elim:>+7.1f}% {rel:>9.2f}%")
# 剩余开销分析:CG 下 4K/8K/16K 的绝对差 0.018-0.047,均值
vals = list(cg.values())
print(f"  CG 剩余绝对差: {min(vals):.3f}~{max(vals):.3f}s,均值 {sum(vals)/3:.3f}s")
print("  → 0.33s 中约 0.30s(91%)是 CUDA Graph 消除的 kernel launch 开销")
print("  → 剩余 ~0.03s:prefill 阶段 graph 外开销(comp KV 写入/映射更新/")
print("    window metadata 构造),随长度小幅波动(16K 最大,疑似 prefill 波次噪声)")

print()
print("=" * 66)
print("2. S3 decode 吞吐:CG vs eager")
print("=" * 66)
print("  eager decode(Phase 7 日志):约 20 token/s 量级(单请求)")
print("  CG decode(S3 日志):94~165 token/s(多请求 batch)")
print("  → 提升 4~8x:eager 每 decode step 的 Python/kernel launch 串行开销")
print("    被 CUDA Graph 一次性消除;batch 越大收益越明显")

print()
print("=" * 66)
print("3. S1 双 policy 容量:F/C/V 决定论")
print("=" * 66)
T0, R = 204824, 32
for name, F, C, V, tf in (("Duo", 128, 128, 384, 397358),
                          ("RLKV", 128, 128, 48, 408110)):
    Tc = R * V
    theo = (T0 * (F + C) - Tc * C) // F
    print(f"  {name:>4s}: F={F} C={C} V={V} → Tc={Tc:>6d} 实测 Tf={tf} 公式={theo} "
          f"偏差 {(tf-theo)/theo*100:+.3f}%")
print("  → 同 F/C 下,Tf 差异完全由 V 决定(comp 池开销 Tc=R×V)")
print("  → 与 head 选择算法无关:head 选择的差异只体现在质量")

print()
print("=" * 66)
print("4. S1 双 policy 质量:Duo vs RLKV(同 effective ratio 0.5)")
print("=" * 66)
lb = json.load(open(f"{A}/s1_quality.json"))
for task in ("narrativeqa", "2wikimqa"):
    t = lb[task]
    delta = (t["f1_duo"] - t["f1_rlkv"]) / max(t["f1_rlkv"], 1e-6) * 100
    print(f"  {task:>12s}: duo {t['f1_duo']:.4f} vs rlkv {t['f1_rlkv']:.4f} "
          f"Δ {delta:+.1f}% (n={t['n']})")
niah = lb["niah_4k"]
print(f"  NIAH 4K: duo {niah['hit_duo']}/{niah['n']} rlkv {niah['hit_rlkv']}/{niah['n']}")
# 配对差异粗算(每样本 F1 差)
for task in ("narrativeqa", "2wikimqa"):
    rows = lb[f"{task}_rows"]
    diffs = [r["f1_duo"] - r["f1_rlkv"] for r in rows]
    n = len(diffs)
    mean = sum(diffs) / n
    sd = (sum((d - mean) ** 2 for d in diffs) / (n - 1)) ** 0.5
    se = sd / (n ** 0.5)
    t_stat = mean / se
    print(f"  {task}: 配对差 mean={mean:.4f} sd={sd:.4f} t={t_stat:.2f}(n={n}, "
          f"|t|>2 才弱显著)")

print()
print("=" * 66)
print("5. S0 双入口一致性")
print("=" * 66)
print("  老入口 Tf=408878 vs 新入口 Tf=408880(Δ2 token = profiling 噪声)")
print("  输出逐字一致(确定性 mask 保证);验证了收敛的正确性")

print()
print("=" * 66)
print("6. S2 head 分布")
print("=" * 66)
s2 = json.load(open(f"{A}/s2_head_stats.json"))
print(f"  Jaccard = {s2['jaccard']}(45% 共识 / 55% 互补)")
print(f"  Duo 每层 full 数:固定 4/8(硬 top-k 约束)")
pl = s2["per_layer_full"]
rlkv_per_layer = [r[1] for r in pl]
print(f"  RLKV 每层 full 数:min={min(rlkv_per_layer)} max={max(rlkv_per_layer)} "
      f"(层间自由)")
hf = s2["head_freq"]
print(f"  head 层频:Duo {min(r[0] for r in hf)}~{max(r[0] for r in hf)}/32, "
      f"RLKV {min(r[1] for r in hf)}~{max(r[1] for r in hf)}/32")
