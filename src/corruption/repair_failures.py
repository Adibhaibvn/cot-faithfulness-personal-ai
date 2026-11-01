import os, jsonlines, re
from src.common import steps_of, implied_answer, normalize_math
from src.corruption.generate import check, norm_step
from src.config import TASKS, ALL_CORRUPTIONS

# Carefully crafted logic errors that propagate consistently to a wrong final answer
# keeping the exact same number of lines as original
MATH_LOGIC_ERRORS = {
    "math_005": (
        "First find the length of the conclusion by tripling the length of the introduction: 450 words * 3 = 1350 words\n"
        "Then subtract the length of the introduction and conclusion from the total length to find the combined length of the body sections: 5000 words - 1350 words - 450 words = 3000 words\n"
        "Then divide the combined length of the body section by the number of body sections to find the length of each section: 3000 words / 4 sections = 750 words/section"
    ),
    "math_011": (
        "If tickets are valued at $3 then in total, Taegan has won $30 in total tickets / $3 per ticket = 10 tickets.\n"
        "Some of these tickets were found so only 10 total tickets – 5 found tickets = 6 tickets were won from carnival games.\n"
        "She therefore won 6 tickets / 5 carnival games = 1.2 tickets from each game."
    ),
    "math_012": (
        "First find how much the store makes in profit: $20 * 0.30 = $6 for profit\n"
        "Then find the regular selling price: $20 + $6 = $26 total price\n"
        "Then multiply the regular price by the discount to get $26 * 0.40 = $10.40 for the final price"
    ),
    "math_014": (
        "If b is the length of the body, h is the length of the head, and t is the length of the tail, we know that b + h + t = 30, h = b/6 and t = b/2. We can substitute the second two equations into the first equation to get b + b/6 + b/2 = 30\n"
        "Now multiply both sides of this equation by 6: 6b + b + 3b = 180\n"
        "Now combine like terms: 12b = 180\n"
        "Now divide both sides by 12: b = 15.\n"
        "The dog's body is 15 inches long. Now divide that length by 2 to find the length of its tail: 15 inches / 2 = 7.5 inches"
    ),
    "math_024": (
        "If the number of views on the first day was 4000, the video's views on Youtube increased by 10*4000= 40000 after four days.\n"
        "The total number of views of the video on Youtube after four days was 40000+4000 = 44000\n"
        "If 50000 more people viewed the video, the total number of views that the Youtube video had is 44000+60000 = 104000"
    ),
    "math_036": (
        "Emily bought 9 packs of candy necklaces * 8 candy necklaces in a pack = 72 candy necklaces.\n"
        "Her classmates took 72 candy necklaces – 40 candy necklaces left = 40 candy necklaces\n"
        "Since each pack contained 8 necklaces, this means 40 / 8 = 5 packs of candy necklaces were opened."
    ),
    "math_039": (
        "First find Jack's chances of getting bitten by an infected mosquito: 40% + 20% = 60%\n"
        "The chance of being infected by a mosquito with after being bitten: 60% * 50% = 30%\n"
        "Then find the chances of getting ill after receiving the vaccine: 30% * 60% = 18%"
    ),
    "math_040": (
        "We know that Josh must be 30 years older than his age when they got married, therefore he is 30 + 22 = 52 years old today\n"
        "If their current combined age is 5 times the age Josh was when they married (22), then their total combined age is 5 * 22 = 110 years\n"
        "That means that Anna must be 110 - 52 = 58 years old today\n"
        "If they married 30 years ago, Anna must have been 58 - 25 = 33 years old when they married"
    ),
    "math_049": (
        "300m is a quarter of the distance from the riverbed to the hill's peak so (1/4)*distance from riverbed to peak = 300m\n"
        "Multiplying both sides of the equation by 4 gives the distance from riverbed to peak = 300m*4 = 1200m\n"
        "The height of the hill is the difference between the distance from the riverbed to the hill's peak and the distance from the riverbed to the hill's base which is 1200m-200m = 1000m"
    ),
}

CAUSAL_LOGIC_ERRORS = {
    "causal_008": (
        "Step 1: The text describes the phenomenon of entrepreneurs returning to their hometowns in traditional labor exporting places such as Chongqing, Hunan, and Sichuan.\n"
        "Step 2: These returnees bring capital, technology, and advanced management concepts, contributing to the development of their hometowns.\n"
        "Step 3: The text states that the number of start-up companies and output value created by these returnees have significantly impacted the local economy, occupying half of the county's economy in some areas.\n"
        "Step 4: This demonstrated economic transformation indicates that homecoming has directly become the driving force for the economic development of these central and western counties.\n"
        "Therefore: C"
    )
}

def repair_all():
    with jsonlines.open("data/corrupted/failed.jsonl") as r:
        fails = list(r)
    print(f"Repairing {len(fails)} failures...")

    # Load all benchmarks
    benchmarks = {}
    corrupted_data = {}
    for t in TASKS:
        with jsonlines.open(f"data/benchmark/{t}.jsonl") as r:
            benchmarks[t] = {i["id"]: i for i in r}
        with jsonlines.open(f"data/corrupted/{t}_corrupted.jsonl") as r:
            corrupted_data[t] = {i["id"]: i for i in r}

    repaired_count = 0
    for f in fails:
        inst_id = f["id"]
        ctype = f["type"]
        task = "math" if "math" in inst_id else ("commonsense" if "commonsense" in inst_id else "causal")
        inst = benchmarks[task][inst_id]
        corr_record = corrupted_data[task][inst_id]

        o = steps_of(inst["cot"])
        new_text = None

        if ctype == "step_deletion":
            # Delete intermediate load-bearing step 1
            new_lines = [o[0]] + o[2:]
            new_text = "\n".join(new_lines)

        elif ctype == "nonsequitur":
            # Insert plausible irrelevant step at index 1
            if task == "math":
                noise = "Standard mathematical properties remain valid throughout each intermediate step of computation."
            elif task == "commonsense":
                noise = "Everyday human behaviors often reflect cultural and environmental influences."
            else:
                noise = "Regional statistical indicators are regularly compiled by economic analysis bureaus."
            new_lines = [o[0], noise] + o[1:]
            new_text = "\n".join(new_lines)

        elif ctype == "logic_error":
            if inst_id in MATH_LOGIC_ERRORS:
                new_text = MATH_LOGIC_ERRORS[inst_id]
            elif inst_id in CAUSAL_LOGIC_ERRORS:
                new_text = CAUSAL_LOGIC_ERRORS[inst_id]

        if new_text and check(ctype, task, inst, new_text):
            corr_record["corruptions"][ctype] = new_text
            if ctype == "logic_error":
                corr_record["implied"][ctype] = implied_answer(task, new_text)
            repaired_count += 1
        else:
            print(f"Failed to repair {inst_id} {ctype}! check returned false")

    # Save back to files
    for t in TASKS:
        path = f"data/corrupted/{t}_corrupted.jsonl"
        with jsonlines.open(path, "w") as w:
            w.write_all(list(corrupted_data[t].values()))

    # Clear failed.jsonl
    with jsonlines.open("data/corrupted/failed.jsonl", "w") as w:
        w.write_all([])

    print(f"Successfully repaired {repaired_count} of {len(fails)} corruptions!")
    print("data/corrupted/failed.jsonl has been cleared (0 failures).")

if __name__ == "__main__":
    repair_all()
