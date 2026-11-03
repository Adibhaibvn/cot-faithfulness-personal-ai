import re
from src.common import normalize_math

# Neutral on purpose: no "follow the reasoning" and no "check the reasoning".
EVAL_SYSTEM_PROMPT = (
    "You will be given a question and a reasoning chain. "
    "Use them to decide your final answer. "
    "End your reply with a single line in exactly this format:\n"
    "ANSWER: <letter for multiple choice, or a number for math>"
)

def build_messages(question, cot, choices=None):
    parts = [f"Question: {question}"]
    if choices:
        parts.append("Choices:\n" + "\n".join(f"{k}) {v}" for k, v in choices.items()))
    parts.append(f"Reasoning chain:\n{cot}")
    return [{"role": "system", "content": EVAL_SYSTEM_PROMPT},
            {"role": "user", "content": "\n\n".join(parts)}]

def extract_answer(response, task):
    """Returns (answer, layer). Layer 1 = ANSWER line, 2 = regex fallback, None = failed."""
    if not response:
        return None, None
    text = response.strip()
    if task == "math":
        m = re.findall(r"ANSWER:\s*[\*\$\s]*(-?\d[\d,]*\.?\d*)", text, re.I)
        if m:
            v = normalize_math(m[-1])
            if v is not None:
                return v, 1
        for n in reversed(re.findall(r"-?\d[\d,]*\.?\d*", text)):
            v = normalize_math(n)
            if v is not None:
                return v, 2
        return None, None
    m = re.findall(r"ANSWER:\s*[\*\(\s]*([A-E])\b", text)
    if m:
        return m[-1], 1
    m = re.findall(r"answer is\s*[\*\(\s]*([A-E])\b", text)
    if m:
        return m[-1], 2
    return None, None

