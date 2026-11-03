import numpy as np
import pandas as pd
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests
from src.config import CORE_CORRUPTIONS, TASKS

def main():
    try:
        ok = pd.read_csv("results/processed/pairs_valid.csv")
    except Exception as e:
        print("No pairs_valid.csv found yet:", e)
        return

    ok = ok[ok["corruption"].isin(CORE_CORRUPTIONS)].copy()
    ok["changed"] = ok["changed"].astype(bool)

    # ---- 4.4: Bootstrap 95% CIs per model x task ----
    def boot(sub, n=2000, seed=0):
        rng = np.random.default_rng(seed)
        groups = [g["changed"].values.astype(float) for _, g in sub.groupby("id")]
        if not groups:
            return (np.nan, np.nan)
        stats = []
        for _ in range(n):
            idx = rng.integers(0, len(groups), len(groups))
            stats.append(np.concatenate([groups[i] for i in idx]).mean())
        return tuple(np.percentile(stats, [2.5, 97.5]))

    rows = []
    for (m, t), sub in ok.groupby(["model", "task"]):
        lo, hi = boot(sub)
        rows.append({"model": m, "task": t, "n": len(sub),
                     "fs_core": sub["changed"].mean(), "ci_low": lo, "ci_high": hi})
    ci_df = pd.DataFrame(rows).round(3)
    ci_df.to_csv("results/processed/fs_ci.csv", index=False)
    print("Bootstrap 95% CIs (fs_ci.csv):")
    print(ci_df.to_string(index=False), "\n")

    # ---- 4.3: Exact McNemar for planned comparisons with Holm correction ----
    COMPARISONS = [
        ("llama3.1-8b-q4", "llama3.1-8b-q8"),       # RQ2 (Quantization)
        ("llama3.1-8b-q4", "llama3.1-8b-cloud"),    # RQ2 (Quantization vs Full)
        ("llama3.1-8b-q8", "llama3.1-8b-cloud"),    # RQ2
        ("llama3.2-3b",    "llama3.1-8b-q4"),       # RQ1 (Size/Scale)
        ("llama3.1-8b-q4", "llama3.3-70b-cloud"),   # RQ3 (Edge vs Cloud)
        ("llama3.1-8b-q4", "gemma2-9b"),            # Architecture
        ("gemma2-9b",      "gemini-flash"),         # RQ3
    ]

    def mcnemar_exact(a, b):
        m = a.merge(b, on=["id", "corruption"], suffixes=("_a", "_b"))
        n10 = int(((m["changed_a"] == True) & (m["changed_b"] == False)).sum())
        n01 = int(((m["changed_a"] == False) & (m["changed_b"] == True)).sum())
        p = binomtest(n10, n10 + n01, 0.5).pvalue if (n10 + n01) > 0 else 1.0
        return len(m), n10, n01, p

    res = []
    for A, B in COMPARISONS:
        for task in ["all"] + TASKS:
            sub = ok if task == "all" else ok[ok["task"] == task]
            a = sub[sub["model"] == A][["id", "corruption", "changed"]]
            b = sub[sub["model"] == B][["id", "corruption", "changed"]]
            if a.empty or b.empty:
                continue
            n, n10, n01, p = mcnemar_exact(a, b)
            res.append({"A": A, "B": B, "task": task, "n_paired": n,
                        "A_only_changed": n10, "B_only_changed": n01, "p": p})
    
    if res:
        res_df = pd.DataFrame(res)
        res_df["p_holm"] = multipletests(res_df["p"], method="holm")[1]
        res_df["significant_0.05"] = res_df["p_holm"] < 0.05
        res_df.round(4).to_csv("results/processed/mcnemar.csv", index=False)
        print("McNemar Paired Tests with Holm Correction (mcnemar.csv):")
        print(res_df.round(4).to_string(index=False))
    else:
        print("McNemar comparisons require at least 2 evaluated models to pair.")

if __name__ == "__main__":
    main()

