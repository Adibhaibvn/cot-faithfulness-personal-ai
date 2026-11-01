import argparse, os, re, time, jsonlines
from src.common import steps_of, implied_answer
from src.config import MIN_INTERVAL
from src.models.clients import call_model

GEN_MODEL = "llama3.3-70b-cloud"

PROMPT = """Write a step-by-step reasoning chain for the question below.
Format: one line per step starting with "Step N:", then a final line "Therefore: <letter>".
Use 3 to 6 steps. Each step must build on the previous one. Output only these lines.

Example:
Question: What do fish need to survive?
Choices:
A) sunlight
B) oxygen
C) soil
Step 1: Fish are aquatic animals that breathe underwater.
Step 2: They extract dissolved oxygen from water using gills.
Step 3: Oxygen is essential for their cellular respiration.
Therefore: B

Question: {question}
Choices:
{choices}
"""

def clean(text):
    lines = [l for l in steps_of(text or "") if re.match(r"(Step \d+:|Therefore:)", l)]
    return "\n".join(lines)

def build(task, n_keep=50):
    cand_path = f"data/benchmark/{task}_candidates.jsonl"
    out_path = f"data/benchmark/{task}.jsonl"

    if not os.path.exists(cand_path):
        raise FileNotFoundError(f"Candidate file not found: {cand_path}")

    with jsonlines.open(cand_path) as r:
        cands = list(r)

    # Resume from existing progress if file exists
    kept = []
    seen_cand_ids = set()
    if os.path.exists(out_path):
        with jsonlines.open(out_path) as r:
            for row in r:
                kept.append(row)
                if "cand_id" in row:
                    seen_cand_ids.add(row["cand_id"])
        print(f"Resuming {task}: {len(kept)} already kept in {out_path}")

    pace = MIN_INTERVAL.get("openrouter", 0.5)
    tried = 0

    for inst in cands:
        if len(kept) >= n_keep:
            break

        cand_id = inst.get("id")
        if cand_id in seen_cand_ids:
            continue

        tried += 1
        choices = "\n".join(f"{k}) {v}" for k, v in inst["choices"].items())
        for temp in (0.2, 0.7):
            text, meta = call_model(GEN_MODEL, [{"role": "user", "content":
                                 PROMPT.format(question=inst["question"], choices=choices)}],
                                 max_tokens=400, temperature=temp)
            if pace:
                time.sleep(pace)

            cot = clean(text)
            n = len(steps_of(cot))
            if 4 <= n <= 8 and implied_answer(task, cot) == inst["answer"]:
                kept_inst = dict(inst, cot=cot, id=f"{task}_{len(kept):03d}", cand_id=cand_id)
                kept.append(kept_inst)
                seen_cand_ids.add(cand_id)
                # Cache immediately to disk
                with jsonlines.open(out_path, mode="a") as w:
                    w.write(kept_inst)
                break

        print(f"{task}: kept {len(kept)}/{n_keep} (tried {tried} new candidates)", end="\r")

    print(f"\n{task}: {len(kept)} total kept from {tried} newly tried candidates")
    return len(kept)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50, help="Number of instances to keep per task")
    parser.add_argument("--task", choices=["commonsense", "causal", "all"], default="all")
    args = parser.parse_args()

    tasks = ["commonsense", "causal"] if args.task == "all" else [args.task]
    for t in tasks:
        build(t, n_keep=args.limit)

