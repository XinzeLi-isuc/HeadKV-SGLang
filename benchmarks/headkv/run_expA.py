"""实验 A:长上下文"有界差异"验证——DuoKV vs FullKV 的分叉率。

主张 3 补充:长于窗口的 prompt,DuoKV 的差异应来自窗口语义(有界、可解释),
而非随机错误。测量:
- 逐 token 一致率(分叉位置分布)
- 首 token 一致率(生成起点未系统性偏离)
- 分叉后是否保持连贯(greedy 输出非乱码)

prompt 构造:有信息的长文本(filler + 随机 facts + 结尾问题),
8K 与 16K 各 5 条。
"""
import argparse
import json
import random
import urllib.request

FILLER = (
    "The history of the ancient kingdom spans many centuries, with its capital "
    "city serving as a crossroads of trade, culture, and scholarship. Scholars "
    "from distant lands gathered in its libraries, documenting laws, poetry, "
    "and the movements of the stars. Each generation added new chapters to the "
    "chronicle, preserving knowledge for those who would come after. "
)
FACT_TMPL = "Fact {i}: The {subject} of {place} was named {name} in the year {year}."


def build_long_prompt(target_tokens, seed):
    rng = random.Random(seed)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        "/home/lixinze/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct"
    )
    filler_ids = tok(FILLER, add_special_tokens=False)["input_ids"]
    ids = []
    while len(ids) < target_tokens:
        ids.extend(filler_ids)
    ids = ids[:target_tokens]
    # 尾部插入 facts(生成起点附近的局部信息,与窗口内信息同源)
    subjects = ["river", "bridge", "temple", "market", "palace", "garden", "harbor", "fortress"]
    places = ["the north valley", "the eastern shore", "the high plateau", "the delta"]
    names = ["Aldric", "Branwen", "Cedric", "Dalia", "Eamon", "Freya"]
    years = [842, 917, 1003, 1156, 1274, 1388]
    fact_ids = []
    for i in range(8):
        f = FACT_TMPL.format(i=i, subject=rng.choice(subjects),
                             place=rng.choice(places),
                             name=rng.choice(names), year=rng.choice(years))
        fact_ids.extend(tok(f + " ", add_special_tokens=False)["input_ids"])
    ids = ids + fact_ids
    q = ("\n\nBased on the text, answer: What is the name of the "
         "subject of place mentioned in Fact 5?")
    q_ids = tok(q, add_special_tokens=False)["input_ids"]
    return tok.decode(ids + q_ids, skip_special_tokens=True)


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

    results = []
    for L in (8192, 16384):
        for seed in range(5):
            text = build_long_prompt(L, seed)
            o_f = gen(30060, text)  # FullKV
            o_d = gen(30061, text)  # DuoKV ratio 0.5
            ids_f = tok(o_f)["input_ids"]
            ids_d = tok(o_d)["input_ids"]
            n = min(len(ids_f), len(ids_d))
            match = sum(1 for i in range(n) if ids_f[i] == ids_d[i])
            first_div = next((i for i in range(n) if ids_f[i] != ids_d[i]), n)
            row = {
                "len": L, "seed": seed,
                "first_tok_match": ids_f[0] == ids_d[0],
                "token_agreement": round(match / n, 4) if n else 1.0,
                "first_divergence_pos": first_div,
                "fullkv_head": o_f.strip()[:50], "duokv_head": o_d.strip()[:50],
            }
            results.append(row)
            print(f"L={L} seed={seed}: first_tok={row['first_tok_match']} "
                  f"agreement={row['token_agreement']} div@pos {first_div}")

    with open("/home/lixinze/HeadKV-SGLang/artifacts/expA_bounded_diff.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    # 汇总
    for L in (8192, 16384):
        rows = [r for r in results if r["len"] == L]
        ft = sum(1 for r in rows if r["first_tok_match"])
        ag = sum(r["token_agreement"] for r in rows) / len(rows)
        print(f"\n[{L}] first-token match {ft}/{len(rows)}; "
              f"avg token agreement {ag:.3f}")


if __name__ == "__main__":
    main()
