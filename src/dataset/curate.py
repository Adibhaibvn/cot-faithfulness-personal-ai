import os, json, random, jsonlines
from datasets import load_dataset
from src.common import steps_of, strip_gsm8k_markup, normalize_math, implied_answer

SEED = 42
os.makedirs("data/benchmark", exist_ok=True)

def curate_math(n=50):
    ds = load_dataset("openai/gsm8k", "main")["train"]
    total = len(ds)
    idx = list(range(total))
    random.Random(SEED).shuffle(idx)
    out = []
    # --- diagnostic counters ---
    rejected = {"gt_none": 0, "step_count": 0, "implied_mismatch": 0}
    scanned = 0
    for i in idx:
        if len(out) == n:
            break
        scanned += 1
        ex = ds[i]
        body, _, final = ex["answer"].partition("####")
        cot = strip_gsm8k_markup(body).strip()
        gt = normalize_math(final)
        steps = steps_of(cot)
        if gt is None:
            rejected["gt_none"] += 1
            continue
        if not (3 <= len(steps) <= 8):
            rejected["step_count"] += 1
            continue
        if implied_answer("math", cot) != gt:      # last line must state the answer
            rejected["implied_mismatch"] += 1
            continue
        out.append({"id": f"math_{len(out):03d}", "task": "math",
                    "question": ex["question"], "choices": None,
                    "cot": cot, "answer": gt})
    with jsonlines.open("data/benchmark/math.jsonl", "w") as f:
        f.write_all(out)
    print(f"math: {len(out)} instances saved")
    print(f"  dataset size: {total}")
    print(f"  scanned: {scanned}")
    print(f"  acceptance rate: {len(out)/max(scanned,1):.1%}")
    print(f"  rejections: {rejected}")

def curate_commonsense_candidates(n=90):
    ds = load_dataset("tau/commonsense_qa")["train"]
    total = len(ds)
    idx = list(range(total))
    random.Random(SEED).shuffle(idx)
    out = []
    rejected = {"empty_answerkey": 0}
    scanned = 0
    for i in idx:
        if len(out) == n:
            break
        scanned += 1
        ex = ds[i]
        if not ex["answerKey"]:
            rejected["empty_answerkey"] += 1
            continue
        choices = dict(zip(ex["choices"]["label"], ex["choices"]["text"]))
        out.append({"id": f"commonsense_c{len(out):03d}", "task": "commonsense",
                    "question": ex["question"], "choices": choices,
                    "answer": ex["answerKey"]})
    with jsonlines.open("data/benchmark/commonsense_candidates.jsonl", "w") as f:
        f.write_all(out)
    print(f"commonsense: {len(out)} candidates saved")
    print(f"  dataset size: {total}")
    print(f"  scanned: {scanned}")
    print(f"  acceptance rate: {len(out)/max(scanned,1):.1%}")
    print(f"  rejections: {rejected}")

def curate_logiqa_candidates(n=90):
    # lucasmccabe/logiqa uses a legacy HF loading script unsupported by datasets>=5.0.
    # Load from the same upstream source (github.com/lgw863/LogiQA-dataset) saved locally.
    with open("data/raw/logiqa_train.json") as f:
        ds = json.load(f)
    total = len(ds)
    labels = ["A", "B", "C", "D"]
    idx = list(range(total))
    random.Random(SEED).shuffle(idx)
    out = []
    rejected = {"not_4_options": 0, "too_long": 0}
    scanned = 0
    for i in idx:
        if len(out) == n:
            break
        scanned += 1
        ex = ds[i]
        if len(ex["options"]) != 4:
            rejected["not_4_options"] += 1
            continue
        q = f"{ex['context']}\n\nQuestion: {ex['query']}"
        if len(q) > 1800:                           # keep inside local context windows
            rejected["too_long"] += 1
            continue
        out.append({"id": f"causal_c{len(out):03d}", "task": "causal",
                    "question": q, "choices": dict(zip(labels, ex["options"])),
                    "answer": labels[ex["correct_option"]]})
    with jsonlines.open("data/benchmark/causal_candidates.jsonl", "w") as f:
        f.write_all(out)
    print(f"causal: {len(out)} candidates saved")
    print(f"  dataset size: {total}")
    print(f"  scanned: {scanned}")
    print(f"  acceptance rate: {len(out)/max(scanned,1):.1%}")
    print(f"  rejections: {rejected}")

if __name__ == "__main__":
    curate_math()
    curate_commonsense_candidates()
    curate_logiqa_candidates()

