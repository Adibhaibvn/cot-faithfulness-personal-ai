import glob, hashlib, jsonlines
from src.config import TASKS, ALL_CORRUPTIONS

total = 0
for t in TASKS:
    path = f"data/corrupted/{t}_corrupted.jsonl"
    with jsonlines.open(path) as r:
        for inst in r:
            assert set(inst["corruptions"]) == set(ALL_CORRUPTIONS), inst["id"]
            assert all(inst["corruptions"].values()), f"missing corruption in {inst['id']}"
            total += 1
assert total == 150, f"expected 150 instances, found {total}"

lines = []
for p in sorted(glob.glob("data/corrupted/*_corrupted.jsonl")):
    lines.append(f"{hashlib.sha256(open(p, 'rb').read()).hexdigest()}  {p}")
open("data/FROZEN_SHA256.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
print("Frozen: 150 instances x 4 corruptions = 600")

