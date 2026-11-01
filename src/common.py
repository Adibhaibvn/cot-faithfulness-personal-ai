import re

def steps_of(cot):
    return [l.strip() for l in cot.split("\n") if l.strip()]

def normalize_math(s):
    """'$1,200.00' -> '1200'; returns None if not numeric."""
    if s is None:
        return None
    s = str(s).replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        f = float(s)
    except ValueError:
        return None
    return str(int(f)) if f == int(f) else str(round(f, 4))

def strip_gsm8k_markup(text):
    return re.sub(r"<<[^>]*>>", "", text)

def implied_answer(task, cot):
    """Answer a chain concludes with, read from its last line."""
    lines = steps_of(cot)
    if not lines:
        return None
    last = lines[-1]
    if task == "math":
        nums = re.findall(r"-?\d[\d,]*\.?\d*", last)
        return normalize_math(nums[-1]) if nums else None
    m = re.search(r"Therefore:\s*\(?([A-E])\b", last)
    return m.group(1) if m else None
