"""对 SGLang server 发单个 generate 请求并保存响应(Phase 0 smoke 用)。

用法:
    python send_generate.py --port 30000 --prompt-file prompt_4k.txt --max-new-tokens 32 \
        --out artifacts/fullkv_smoke_resp.json
"""
import argparse
import json
import time
import urllib.request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.prompt_file) as f:
        text = f.read()

    payload = {
        "text": text,
        "sampling_params": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": 0.0,
        },
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{args.port}/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as resp:
        out = json.loads(resp.read().decode())
    dt = time.time() - t0

    result = {
        "port": args.port,
        "prompt_file": args.prompt_file,
        "prompt_tokens": len(text.split()),  # 近似;精确值见 server 日志
        "max_new_tokens": args.max_new_tokens,
        "elapsed_s": round(dt, 3),
        "text": out.get("text", ""),
        "meta": out.get("meta", {}),
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps({"elapsed_s": round(dt, 3), "output": result["text"][:200],
                      "meta": result["meta"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
