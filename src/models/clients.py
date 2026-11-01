import os, time
from dotenv import load_dotenv
from openai import OpenAI
from src.config import MODELS

load_dotenv()
_ollama = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
_or = None
_gem = None

def _openrouter():
    global _or
    if _or is None:
        _or = OpenAI(api_key=os.getenv("OPENROUTER_API_KEY"),
                     base_url="https://openrouter.ai/api/v1")
    return _or

def _gemini():
    global _gem
    if _gem is None:
        from google import genai
        _gem = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    return _gem

def _call_once(cfg, messages, max_tokens, temperature):
    p = cfg["provider"]
    if p == "ollama":
        r = _ollama.chat.completions.create(
            model=cfg["id"], messages=messages, temperature=temperature,
            max_tokens=max_tokens, seed=0)
        return r.choices[0].message.content, {}
    if p == "openrouter":
        kwargs = {}
        if cfg.get("quantizations"):
            kwargs["extra_body"] = {"provider": {"quantizations": cfg["quantizations"]}}
        r = _openrouter().chat.completions.create(
            model=cfg["id"], messages=messages, temperature=temperature,
            max_tokens=max_tokens, **kwargs)
        return r.choices[0].message.content, {"provider": getattr(r, "provider", None)}
    if p == "gemini":
        from google.genai import types
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        user = "\n\n".join(m["content"] for m in messages if m["role"] != "system")
        r = _gemini().models.generate_content(
            model=cfg["id"], contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system or None, temperature=temperature,
                max_output_tokens=max(max_tokens, 1024)))
        return r.text, {}
    raise ValueError(f"unknown provider {p}")

def call_model(model_key, messages, max_tokens=300, temperature=0.0, retries=5):
    cfg = MODELS[model_key]
    last = None
    for attempt in range(retries):
        try:
            t0 = time.time()
            text, meta = _call_once(cfg, messages, max_tokens, temperature)
            meta["latency_s"] = round(time.time() - t0, 2)
            return text, meta
        except Exception as e:
            last = str(e)[:300]
            time.sleep(min(60, 2 ** (attempt + 1)))
    return None, {"error": last}
