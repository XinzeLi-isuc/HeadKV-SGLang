"""Experiment 5: HeadKV DuoPolicy LongBench —— 官方协议版。

与官方 pred.py 完全同一协议(prompt 模板 / max_gen / post_process),
只是生成端是 HeadKV SGLang server(fullkv 或 headkv-duo),保证
Full Attention / Official Duo / HeadKV 三系统公平对比。

用法: python exp5_headkv_longbench.py <port> <tag>
输出: artifacts/exp5_headkv_<tag>_<task>.jsonl(官方 eval.py 可读格式)
"""
import json
import sys
import urllib.request

from datasets import load_dataset

CFG = "/home/lixinze/duo-attention-ref/eval/LongBench/config"
OUT = "/home/lixinze/HeadKV-SGLang/artifacts"


def post_process(response, model_name):
    if "llama-3" in model_name.lower():
        response = (
            response.split(".assistant")[0]
            .split("\n\nQuestion")[0]
            .split("</s>")[0]
            .strip()
        )
    return response


def gen(port, text, max_new):
    payload = {"text": text, "sampling_params": {"max_new_tokens": max_new,
                                                 "temperature": 0.0}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/generate",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as resp:
        return json.loads(resp.read().decode())["text"]


def main():
    port, tag = int(sys.argv[1]), sys.argv[2]
    dataset2prompt = json.load(open(f"{CFG}/dataset2prompt.json"))
    dataset2maxlen = json.load(open(f"{CFG}/dataset2maxlen.json"))
    model_name = "Meta-Llama-3.1-8B-Instruct"
    for task in ["narrativeqa", "2wikimqa"]:
        data = load_dataset("THUDM/LongBench", task, split="test").select(range(30))
        prompt_format = dataset2prompt[task]
        max_gen = dataset2maxlen[task]
        preds = []
        for i, ex in enumerate(data):
            prompt = prompt_format.format(**ex)
            raw = gen(port, prompt, max_gen)
            pred = post_process(raw, model_name)
            preds.append({"pred": pred, "answers": ex["answers"],
                          "all_classes": ex["all_classes"],
                          "length": ex.get("length")})
            print(f"[{task} {i}] pred={pred[:50]!r}")
        path = f"{OUT}/exp5_headkv_{tag}_{task}.jsonl"
        with open(path, "w") as f:
            for p in preds:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"saved -> {path}")


if __name__ == "__main__":
    main()
