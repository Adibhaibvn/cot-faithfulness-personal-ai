import argparse, os, random, re, time, jsonlines
from src.common import steps_of, implied_answer, normalize_math
from src.models.clients import call_model
from src.corruption.prompts import CORRUPTION_PROMPTS
from src.config import TASKS, MIN_INTERVAL, ALL_CORRUPTIONS

def choices_block(choices):
    if not choices:
        return ""
    return "Choices:\n" + "\n".join(f"{k}) {v}" for k, v in choices.items())

def swap_conclusion(task, inst):
    """Rule-based: replace the final answer with a plausible wrong one."""
    rng = random.Random(inst["id"])
    lines = steps_of(inst["cot"])
    last = lines[-1]
    if task == "math":
        gold = float(inst["answer"])
        for _ in range(20):
            wrong = gold + rng.choice([-10, -5, -2, -1, 1, 2, 5, 10])
            if wrong != gold and wrong >= 0:
                break
        wrong_s = str(int(wrong)) if wrong == int(wrong) else str(round(wrong, 4))
        spans = list(re.finditer(r"-?\d[\d,]*\.?\d*", last))
        if not spans:
            return None
        s = spans[-1]
        new_last = last[:s.start()] + wrong_s + last[s.end():]
    else:
        wrong = rng.choice([k for k in inst["choices"] if k != inst["answer"]])
        new_last = re.sub(r"(Therefore:\s*)\(?[A-E]\b\)?", rf"\g<1>{wrong}", last)
    return "\n".join(lines[:-1] + [new_last])

def norm_step(s):
    """Strip leading 'Step N:' prefix to compare step content accurately."""
    return re.sub(r"^Step \d+:\s*", "", s.strip())

def check(ctype, task, inst, new):
    o, c = steps_of(inst["cot"]), steps_of(new or "")
    if not c or (new or "").strip() == inst["cot"].strip():
        return False
    imp = implied_answer(task, new)
    o_norm = [norm_step(x) for x in o]
    c_norm = [norm_step(x) for x in c]
    min_steps = 2 if task == "math" else 3

    if ctype == "logic_error":
        return len(c) == len(o) and c_norm[-1] != o_norm[-1] and imp is not None and imp != inst["answer"]
    if ctype == "step_deletion":
        return len(c) == len(o) - 1 and len(c) >= min_steps and c_norm[-1] == o_norm[-1]
    if ctype == "nonsequitur":
        return len(c) == len(o) + 1 and c_norm[0] == o_norm[0] and c_norm[-1] == o_norm[-1]
    return False

def clean(text):
    text = re.sub(r"```\w*", "", text or "").strip()
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if lines and (lines[0].lower().startswith("here is") or lines[0].lower().startswith("modified")):
        lines.pop(0)
    return "\n".join(lines)

def process(task, model, limit=None):
    bench_path = f"data/benchmark/{task}.jsonl"
    out_path = f"data/corrupted/{task}_corrupted.jsonl"
    os.makedirs("data/corrupted", exist_ok=True)

    with jsonlines.open(bench_path) as r:
        insts = list(r)

    if limit:
        insts = insts[:limit]

    # Load existing progress for resumption
    completed = {}
    if os.path.exists(out_path):
        with jsonlines.open(out_path) as r:
            for row in r:
                completed[row["id"]] = row

    pace = MIN_INTERVAL.get("openrouter", 0.5)
    out, failed = [], []

    for n, inst in enumerate(insts, 1):
        inst_id = inst["id"]
        existing = completed.get(inst_id, {})
        corr = dict(existing.get("corruptions", {}))
        implied = dict(existing.get("implied", {}))
        attempts = dict(existing.get("attempts", {}))

        # Check if all 4 corruptions are already present and valid
        if set(corr.keys()) == set(ALL_CORRUPTIONS) and all(corr.values()):
            out.append(existing)
            print(f"{task}: {n}/{len(insts)} (cached)", end="\r")
            continue

        for ctype, template in CORRUPTION_PROMPTS.items():
            if corr.get(ctype):
                continue
            prompt = template.format(question=inst["question"],
                                     choices=choices_block(inst.get("choices")),
                                     answer=inst["answer"], cot=inst["cot"])
            corr[ctype] = None
            attempts[ctype] = 0
            for temp in (0.3, 0.7, 1.0):
                attempts[ctype] += 1
                text, _ = call_model(model, [{"role": "user", "content": prompt}],
                                     max_tokens=500, temperature=temp)
                if pace:
                    time.sleep(pace)
                text = clean(text)
                if check(ctype, task, inst, text):
                    corr[ctype] = text
                    break
            if ctype == "logic_error" and corr[ctype]:
                implied[ctype] = implied_answer(task, corr[ctype])
            if corr[ctype] is None:
                failed.append({"id": inst["id"], "type": ctype})

        if not corr.get("conclusion_swap"):
            sw = swap_conclusion(task, inst)
            corr["conclusion_swap"] = sw
            implied["conclusion_swap"] = implied_answer(task, sw) if sw else None

        inst_corrupted = dict(inst, corruptions=corr, implied=implied, attempts=attempts)
        out.append(inst_corrupted)
        completed[inst_id] = inst_corrupted

        # Save immediately to disk so progress is preserved
        with jsonlines.open(out_path, "w") as f:
            f.write_all(list(completed.values()))

        print(f"{task}: {n}/{len(insts)}", end="\r")

    print(f"\n{task}: done, {len(failed)} corruptions failed checks")
    return failed

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.3-70b-cloud",
                    help="generator; use llama3.1-8b-q4 or gemma2-9b for fully local")
    ap.add_argument("--task", default="all", choices=["all", "math", "commonsense", "causal"])
    ap.add_argument("--limit", type=int, default=None, help="Limit number of instances for testing")
    args = ap.parse_args()

    tasks = TASKS if args.task == "all" else [args.task]
    all_failed = []
    for t in tasks:
        all_failed += process(t, args.model, limit=args.limit)

    with jsonlines.open("data/corrupted/failed.jsonl", "w") as f:
        f.write_all(all_failed)
    print(f"Total failed: {len(all_failed)} (see data/corrupted/failed.jsonl)")
