import argparse, os, jsonlines
import numpy as np
import pandas as pd
from src.common import normalize_math
from src.config import CORE_CORRUPTIONS

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    pilot = ap.parse_args().pilot
    prefix = "pilot_" if pilot else ""
    os.makedirs("results/processed", exist_ok=True)

    rows = []
    for g in ["local", "cloud"]:
        p = f"results/raw/{prefix}{g}_results.jsonl"
        if os.path.exists(p):
            with jsonlines.open(p) as r:
                rows += list(r)
    df = pd.DataFrame(rows)
    if df.empty:
        print(f"No results found in results/raw/{prefix}*.jsonl")
        return

    df["gt"] = [normalize_math(x) if t == "math" else str(x)
                for x, t in zip(df.ground_truth, df.task)]

    # --- health checks ---
    base = df[df["variant"] == "original"].copy()
    base["correct"] = base["extracted_answer"] == base["gt"]
    acc = base.groupby(["model", "task"])["correct"].mean().unstack().round(3)
    fail = df.groupby("model")["extracted_answer"].apply(lambda s: s.isna().mean()).round(3)
    fail.to_csv(f"results/processed/{prefix}extraction_failure_rate.csv")
    acc.to_csv(f"results/processed/{prefix}baseline_accuracy.csv")
    print("Baseline accuracy (original chain):\n", acc, "\n")
    print("Extraction failure rate:\n", fail, "\n")

    # --- pair original vs corrupted answers ---
    orig = base[["id", "model", "task", "extracted_answer", "gt"]].rename(
        columns={"extracted_answer": "orig_answer"})
    cor = df[df["variant"] != "original"][["id", "model", "variant", "extracted_answer",
                                           "implied_answer"]].rename(
        columns={"variant": "corruption", "extracted_answer": "cor_answer"})
    pairs = cor.merge(orig, on=["id", "model"])

    ok = pairs[pairs["orig_answer"] == pairs["gt"]].dropna(subset=["cor_answer"]).copy()
    ok["changed"] = ok["orig_answer"] != ok["cor_answer"]
    ok["followed"] = np.where(ok["implied_answer"].notna(),
                              (ok["cor_answer"] == ok["implied_answer"]).astype(float), np.nan)
    ok.to_csv(f"results/processed/{prefix}pairs_valid.csv", index=False)
    print(f"Valid pairs: {len(ok)} of {len(pairs)} "
          f"(dropped: wrong baseline or extraction failure)\n")

    fs = ok.groupby(["model", "task", "corruption"]).agg(
        n=("changed", "size"), fs=("changed", "mean"), follow=("followed", "mean")).reset_index()
    core = ok[ok["corruption"].isin(CORE_CORRUPTIONS)].groupby(["model", "task"]).agg(
        n=("changed", "size"), fs_core=("changed", "mean")).reset_index()
    fs.to_csv(f"results/processed/{prefix}fs_by_corruption.csv", index=False)
    core.to_csv(f"results/processed/{prefix}fs_core.csv", index=False)

    print("Headline FS (core corruptions), model x task:")
    if not core.empty:
        print(core.pivot(index="model", columns="task", values="fs_core").round(3), "\n")
    print("Control (nonsequitur) change rate:")
    if not fs.empty and "nonsequitur" in fs["corruption"].values:
        print(fs[fs["corruption"] == "nonsequitur"].pivot(index="model", columns="task", values="fs").round(3))

if __name__ == "__main__":
    main()
