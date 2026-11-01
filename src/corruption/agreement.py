import numpy as np
import pandas as pd

def cohen_kappa(y1, y2):
    try:
        from sklearn.metrics import cohen_kappa_score
        return cohen_kappa_score(y1, y2)
    except ImportError:
        y1 = np.asarray(y1)
        y2 = np.asarray(y2)
        po = np.mean(y1 == y2)
        p1_1 = np.mean(y1 == 1)
        p2_1 = np.mean(y2 == 1)
        p1_0 = np.mean(y1 == 0)
        p2_0 = np.mean(y2 == 0)
        pe = (p1_1 * p2_1) + (p1_0 * p2_0)
        return (po - pe) / (1.0 - pe) if pe < 1.0 else 1.0

a = pd.read_csv("review/ratings_aditya.csv")[["row_id", "corruption", "valid"]]
v = pd.read_csv("review/ratings_varun.csv")[["row_id", "valid"]]
m = a.merge(v, on="row_id", suffixes=("_a", "_v"))

kappa = cohen_kappa(m.valid_a, m.valid_v)
print(f"Cohen's kappa: {kappa:.3f}  (target > 0.7)")
print(f"Raw agreement: {(m.valid_a == m.valid_v).mean():.1%}")
m["both_valid"] = (m.valid_a == 1) & (m.valid_v == 1)
print("\nValid-by-both rate per corruption type:")
print(m.groupby("corruption")["both_valid"].mean().round(2))
print("\nDisagreements (discuss these):")
disagreements = m[m.valid_a != m.valid_v][["row_id", "corruption"]]
if len(disagreements) == 0:
    print("None")
else:
    print(disagreements.to_string(index=False))
