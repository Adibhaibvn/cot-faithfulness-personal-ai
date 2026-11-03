import argparse, re, jsonlines
from src.common import normalize_math
from src.models.clients import call_model

PROMPT = ("Here is a model's reply to a question. What final answer does it commit to? "
          "Reply with only the letter (A-E) or the number, or UNCLEAR.\n\nReply:\n{r}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--model", default="llama3.1-8b-q4")   # Cloud alternative: llama3.1-8b-cloud
    a = ap.parse_args()

    with jsonlines.open(a.file) as r:
        rows = list(r)
    fixed = 0
    for row in rows:
        if row["extracted_answer"] is None and row.get("response"):
            out, _ = call_model(a.model, [{"role": "user", "content": PROMPT.format(r=row["response"])}],
                                max_tokens=10)
            out = (out or "").strip()
            val = normalize_math(out) if row["task"] == "math" else (out if re.fullmatch(r"[A-E]", out) else None)
            if val:
                row["extracted_answer"], row["layer"] = val, 3
                fixed += 1
    with jsonlines.open(a.file, "w") as f:
        f.write_all(rows)
    print(f"Recovered {fixed} answers in {a.file}")

if __name__ == "__main__":
    main()
