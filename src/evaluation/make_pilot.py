import jsonlines
from src.config import TASKS

def main():
    for t in TASKS:
        with jsonlines.open(f"data/corrupted/{t}_corrupted.jsonl") as r:
            rows = list(r)[:5]
        with jsonlines.open(f"data/corrupted/{t}_pilot.jsonl", "w") as f:
            f.write_all(rows)
    print("Pilot sets written (5 instances per task = 15)")

if __name__ == "__main__":
    main()
