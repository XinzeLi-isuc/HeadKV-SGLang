"""S5: current main 单请求 E2E 开销 —— DuoKV-cg(30091) vs DuoKV-eager(30092)。

口径对齐 S3: 4K prompt, 3-run median 绝对差(t_duo - t_full 不再适用,
此处 duo 的 eager vs cg 自身对比, 输出: median_e2e_cg / median_e2e_eager / 差值)。
"""
import json
import statistics
import time
import urllib.request

PROMPT = open("/home/lixinze/HeadKV-SGLang/artifacts/prompt_4k.txt").read()
OUT = "/home/lixinze/HeadKV-SGLang/artifacts/s5_e2e_overhead_main.json"


def gen(port, text, n=64):
    payload = {"text": text, "sampling_params": {"max_new_tokens": n, "temperature": 0.0}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/generate",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as resp:
        out = json.loads(resp.read().decode())["text"]
    return out, time.time() - t0


def main():
    print(f"prompt tokens: {len(PROMPT.split())} words (4K 上下文模板)")
    cg, eager = [], []
    for r in range(3):
        o_cg, t_cg = gen(30091, PROMPT)
        o_eg, t_eg = gen(30092, PROMPT)
        cg.append(t_cg)
        eager.append(t_eg)
        print(f"run{r}: cg={t_cg:.3f}s eager={t_eg:.3f}s "
              f"diff={t_cg - t_eg:+.3f}s | cg_out={o_cg[:30]!r}")
    m_cg = statistics.median(cg)
    m_eg = statistics.median(eager)
    result = {
        "mode": "current-main duo eager vs cuda-graph",
        "prompt": "prompt_4k.txt",
        "runs": {"cg": cg, "eager": eager},
        "median_e2e_cg_s": m_cg,
        "median_e2e_eager_s": m_eg,
        "cg_minus_eager_s": m_cg - m_eg,
    }
    print(f"\n=== 单请求 E2E(3-run median)===")
    print(f"DuoKV-cg    : {m_cg:.3f}s")
    print(f"DuoKV-eager : {m_eg:.3f}s")
    print(f"cg - eager  : {m_cg - m_eg:+.3f}s")
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
