import csv, os, random, jsonlines
from src.config import TASKS

os.makedirs("review", exist_ok=True)
random.seed(7)
rows = []
for t in TASKS:
    path = f"data/corrupted/{t}_corrupted.jsonl"
    if not os.path.exists(path):
        continue
    with jsonlines.open(path) as r:
        for inst in r:
            for ctype, text in inst["corruptions"].items():
                rows.append({"id": inst["id"], "task": t, "corruption": ctype,
                             "question": inst["question"], "original_cot": inst["cot"],
                             "corrupted_cot": text})

if len(rows) >= 90:
    sample = random.sample(rows, 90)
    with open("review/corruption_sample.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["row_id", "id", "task", "corruption",
                                          "question", "original_cot", "corrupted_cot", "valid"])
        w.writeheader()
        for i, r in enumerate(sample):
            w.writerow({"row_id": i, **r, "valid": ""})
    print("Wrote review/corruption_sample.csv (90 rows)")
else:
    print(f"Only {len(rows)} corruptions available, need 90 to sample.")

gold = []
for t in ["commonsense", "causal"]:
    path = f"data/benchmark/{t}.jsonl"
    if os.path.exists(path):
        with jsonlines.open(path) as r:
            items = list(r)
            gold += random.sample(items, min(10, len(items)))

with open("review/gold_cot_sample.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["id", "task", "question", "choices", "cot",
                                      "gold_answer", "sound"])
    w.writeheader()
    for g in gold:
        w.writerow({"id": g["id"], "task": g["task"], "question": g["question"],
                    "choices": g.get("choices"), "cot": g["cot"],
                    "gold_answer": g["answer"], "sound": ""})
print(f"Wrote review/gold_cot_sample.csv ({len(gold)} rows)")

