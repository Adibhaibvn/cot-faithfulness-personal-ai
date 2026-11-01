import pandas as pd
from sklearn.metrics import cohen_kappa_score

a = pd.read_csv("review/ratings_aditya.csv")[["row_id", "corruption", "valid"]]
v = pd.read_csv("review/ratings_varun.csv")[["row_id", "valid"]]
m = a.merge(v, on="row_id", suffixes=("_a", "_v"))
print(f"Cohen's kappa: {cohen_kappa_score(m.valid_a, m.valid_v):.3f}  (target > 0.7)")
print(f"Raw agreement: {(m.valid_a == m.valid_v).mean():.1%}")
m["both_valid"] = (m.valid_a == 1) & (m.valid_v == 1)
print("\nValid-by-both rate per corruption type:")
print(m.groupby("corruption")["both_valid"].mean().round(2))
print("\nDisagreements (discuss these):")
print(m[m.valid_a != m.valid_v][["row_id", "corruption"]].to_string(index=False))

