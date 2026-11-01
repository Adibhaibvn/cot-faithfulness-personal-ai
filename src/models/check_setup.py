import os, requests
from dotenv import load_dotenv
from src.config import MODELS
load_dotenv()

print("== Ollama ==")
try:
    tags = requests.get("http://localhost:11434/api/tags", timeout=5).json()["models"]
    have = {t["name"] for t in tags}
    for k, v in MODELS.items():
        if v["provider"] == "ollama":
            print("OK     " if v["id"] in have else "MISSING", v["id"])
except Exception as e:
    print("Ollama not reachable:", e)

print("\n== OpenRouter pricing (USD per token; '0' means free) ==")
try:
    data = requests.get("https://openrouter.ai/api/v1/models", timeout=30).json()["data"]
    price = {m["id"]: m["pricing"] for m in data}
    for k, v in MODELS.items():
        if v["provider"] == "openrouter":
            print(v["id"], "->", price.get(v["id"], "NOT LISTED"))
except Exception as e:
    print("OpenRouter check failed:", e)

print("\n== Gemini models visible to your key ==")
try:
    from google import genai
    c = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    names = [m.name for m in c.models.list()]
    print([n for n in names if "gemini" in n])
except Exception as e:
    print("Gemini check failed:", e)

print("\n== Smoke test (one call per model) ==")
from src.models.clients import call_model
msgs = [{"role": "user", "content": "Reply with exactly: ANSWER: B"}]
for k, v in MODELS.items():
    if v["provider"] == "gemini" and not v["id"]:
        print(f"{k:20s} SKIPPED (set GEMINI_MODEL in .env)")
        continue
    text, meta = call_model(k, msgs, max_tokens=20)
    print(f"{k:20s} {repr(text)[:50]}  {meta}")
