import os
from dotenv import load_dotenv
load_dotenv()

MODELS = {
    "llama3.2-3b":        {"group": "local", "provider": "ollama", "id": "llama3.2:3b"},
    "llama3.1-8b-q4":     {"group": "local", "provider": "ollama", "id": "llama3.1:8b"},
    "llama3.1-8b-q8":     {"group": "local", "provider": "ollama", "id": "llama3.1:8b-instruct-q8_0"},
    "gemma2-9b":          {"group": "local", "provider": "ollama", "id": "gemma2:9b"},
    # Ask OpenRouter to route only to providers serving bf16/fp16 weights.
    # If this errors with "no providers", remove "quantizations" and label the
    # model "provider-served, precision unspecified" in the paper.
    "llama3.1-8b-cloud":  {"group": "cloud", "provider": "openrouter",
                           "id": "meta-llama/llama-3.1-8b-instruct",
                           "quantizations": ["bf16", "fp16"]},
    "llama3.3-70b-cloud": {"group": "cloud", "provider": "openrouter",
                           "id": "meta-llama/llama-3.3-70b-instruct"},
    "gemini-flash":       {"group": "cloud", "provider": "gemini",
                           "id": os.getenv("GEMINI_MODEL", "")},
}

TASKS = ["math", "commonsense", "causal"]
CORE_CORRUPTIONS = ["logic_error", "step_deletion", "conclusion_swap"]
CONTROL_CORRUPTIONS = ["nonsequitur"]
ALL_CORRUPTIONS = CORE_CORRUPTIONS + CONTROL_CORRUPTIONS

MIN_INTERVAL = {                      # seconds between calls, per provider
    "ollama": 0.0,
    "openrouter": 0.5,
    "gemini": float(os.getenv("GEMINI_MIN_INTERVAL", "4.5")),
}
