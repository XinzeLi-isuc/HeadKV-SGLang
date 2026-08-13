"""对照:FullKV vs DuoKV(ratio=1.0,全 full 无双池)对分叉 prompts 的输出。"""
import json
import urllib.request

PORTS = {"fullkv": 30000, "duo_r10": 30002}


def gen(port, text):
    payload = {"text": text, "sampling_params": {"max_new_tokens": 32, "temperature": 0.0}}
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/generate",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode())["text"]


with open("/home/lixinze/HeadKV-SGLang/benchmarks/headkv/prompts_correctness.jsonl") as f:
    prompts = {json.loads(l)["id"]: json.loads(l)["prompt"] for l in f}

diverged = ["p02", "p03", "p05", "p08", "p16", "p18", "p20"]
for pid in diverged:
    text = prompts[pid]
    o_f = gen(PORTS["fullkv"], text)
    o_d = gen(PORTS["duo_r10"], text)
    print(f"[{pid}] fullkv==duo(r1.0): {o_f == o_d}")
    if o_f != o_d:
        print(f"  fullkv : {o_f[:80]!r}")
        print(f"  duo r1 : {o_d[:80]!r}")
