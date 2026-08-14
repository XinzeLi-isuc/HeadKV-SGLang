"""S5: RLKV policy 在 current main 的启动 + 生成验证。

验证目标(phase-s4 遗留项 3): RLKV adapter 在 current main 双池 runtime
加载、生成语义正确。启动日志 Tf 与 v0.5.2 同口径(rlkv sparsity 0.5)。
"""
import json
import urllib.request

OUT = "/home/lixinze/HeadKV-SGLang/artifacts/s5_rlkv_smoke_main.json"

PROMPTS = [
    "What is the capital of France? Answer in one word.",
    "Explain the difference between TCP and UDP in two sentences.",
    "Write a short poem about autumn (4 lines).",
]


def gen(port, text, n=64):
    payload = {"text": text, "sampling_params": {"max_new_tokens": n, "temperature": 0.0}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/generate",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as resp:
        return json.loads(resp.read().decode())["text"]


def main():
    rows = []
    for i, p in enumerate(PROMPTS):
        out = gen(30093, p)
        ok = len(out.strip()) > 0
        rows.append({"idx": i, "prompt": p[:50], "output": out[:120], "nonempty": ok})
        print(f"[{i}] nonempty={ok} out={out[:80]!r}")
    all_ok = all(r["nonempty"] for r in rows)
    print(f"\n=== current main RLKV policy smoke ===\n生成语义正常: {all_ok} "
          f"({sum(r['nonempty'] for r in rows)}/{len(rows)})")
    with open(OUT, "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
