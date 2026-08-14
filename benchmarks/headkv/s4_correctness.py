"""S4 验证:current main 上 HeadKV(30080)vs FullKV-triton(30081)的 20 prompts 一致性。"""
import json
import urllib.request


def gen(port, text, n=64):
    payload = {"text": text, "sampling_params": {"max_new_tokens": n, "temperature": 0.0}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/generate",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as resp:
        return json.loads(resp.read().decode())["text"]


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        "/home/lixinze/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct"
    )
    prompts = [json.loads(l)["prompt"] for l in open(
        "/home/lixinze/HeadKV-SGLang/benchmarks/headkv/prompts_correctness.jsonl"
    )]

    exact = 0
    first_tok = 0
    rows = []
    for i, p in enumerate(prompts):
        o_f = gen(30081, p)
        o_d = gen(30080, p)
        ids_f = tok(o_f)["input_ids"]
        ids_d = tok(o_d)["input_ids"]
        n = min(len(ids_f), len(ids_d))
        same = ids_f[:n] == ids_d[:n]
        ft = ids_f[0] == ids_d[0] if n > 0 else False
        exact += same
        first_tok += ft
        rows.append({"idx": i, "exact": same, "first_tok": ft,
                     "fullkv": o_f[:60], "headkv": o_d[:60]})
        print(f"[{i:02d}] exact={same} first_tok={ft}")

    print(f"\n=== current main: HeadKV vs FullKV-triton ===")
    print(f"exact 逐 token 一致: {exact}/{len(prompts)}")
    print(f"首 token 一致: {first_tok}/{len(prompts)}")
    with open("/home/lixinze/HeadKV-SGLang/artifacts/s4_correctness_main.json", "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
