import argparse, json, os, time, jsonlines
from tqdm import tqdm
from src.config import MODELS, TASKS, MIN_INTERVAL
from src.models.clients import call_model
from src.evaluation.extraction import build_messages, extract_answer

def done_keys(path):
    keys = set()
    if os.path.exists(path):
        with jsonlines.open(path) as r:
            for row in r:
                keys.add((row["id"], row["model"], row["variant"]))
    return keys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, choices=["local", "cloud"])
    ap.add_argument("--models", nargs="*", help="subset of model keys")
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()

    keys = [k for k, v in MODELS.items()
            if v["group"] == args.group and (not args.models or k in args.models)]
    out = f"results/raw/{'pilot_' if args.pilot else ''}{args.group}_results.jsonl"
    suffix = "pilot" if args.pilot else "corrupted"
    os.makedirs("results/raw", exist_ok=True)
    done = done_keys(out)
    print(f"Models: {keys} | output: {out} | already done: {len(done)}")

    for mk in keys:
        pace = MIN_INTERVAL[MODELS[mk]["provider"]]
        api_fail = 0
        for task in TASKS:
            with jsonlines.open(f"data/corrupted/{task}_{suffix}.jsonl") as r:
                insts = list(r)
            for inst in tqdm(insts, desc=f"{mk} | {task}"):
                variants = {"original": inst["cot"], **inst["corruptions"]}
                for vname, cot in variants.items():
                    key = (inst["id"], mk, vname)
                    if key in done:
                        continue
                    msgs = build_messages(inst["question"], cot, inst.get("choices"))
                    text, meta = call_model(mk, msgs, max_tokens=300)
                    if text is None:
                        api_fail += 1          # not saved, so a rerun retries it
                        continue
                    ans, layer = extract_answer(text, task)
                    row = {"id": inst["id"], "task": task, "model": mk, "variant": vname,
                           "ground_truth": inst["answer"],
                           "implied_answer": inst.get("implied", {}).get(vname),
                           "response": text, "extracted_answer": ans, "layer": layer, **meta}
                    with open(out, "a") as f:
                        f.write(json.dumps(row) + "\n")
                    done.add(key)
                    if pace:
                        time.sleep(pace)
        print(f"{mk}: finished, API failures this run: {api_fail} (rerun to retry them)")

if __name__ == "__main__":
    main()

